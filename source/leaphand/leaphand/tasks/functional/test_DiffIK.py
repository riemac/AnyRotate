# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

此版本为针对LeapHand机械手的修改版，原版本见scripts/tutorials/05_controllers/run_diff_ik.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from leaphand.robots.leap import LEAP_HAND_CFG  # isort:skip

@configclass
class LeapHandSceneCfg(InteractiveSceneCfg):
    """Configuration for a LeapHand scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
    #     ),
    # )

    # articulation
    robot = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def _convert_Xform_to_body_frame(robot, set_goal, body_name, set_Xform_prim_path):
    """Convert the given set of goals in the set_Xform_prim_path frame to the specified body frame.

    该方法计算Xform标记点相对于目标刚体的固定偏移(只计算一次)，然后将所有目标位姿
    从Xform坐标系转换到刚体坐标系。

    Args:
        robot: The robot instance.
        set_goal: List of goals in the set_Xform_prim_path frame. 每个目标为[x,y,z,qw,qx,qy,qz]格式。
        body_name: The name of the body to convert to (e.g., "fingertip").
        set_Xform_prim_path: The prim path of the frame in which the goals are defined 
                             (e.g., "/World/envs/env_0/Robot/fingertip/index_tip_head").
    Returns:
        List of goals in the specified body frame (same format as input).
        
    Note
    ----
    坐标转换公式
      已知: 
        - goal_marker_in_env: Marker在环境坐标系下的目标位姿 (用户提供)
        - T_marker_in_body: Marker相对于Body的固定偏移 (通过resolve_prim_pose计算)
      求: goal_body_in_env (Body在环境坐标系下的目标位姿)
      
      转换关系(齐次变换):
        T_marker_in_env = T_body_in_env × T_marker_in_body
        
      反向求解:
        T_body_in_env = T_marker_in_env × T_marker_in_body^{-1}
        
      使用 IsaacLab 工具:
        1. 计算逆偏移: offset_inv = (quat_inv(offset_quat), -R_inv * offset_pos)
        2. 应用变换: combine_frame_transforms(marker_in_env, offset_inv)
        
      或者使用 subtract_frame_transforms 的技巧:
        subtract(t01, q01, t02, q02) = (t12, q12)
        其中 T12 = T01^{-1} × T02
        
        我们需要: T_body = T_marker × T_offset^{-1}
        设: T_offset_inv × T_body = T_marker
        即: T_body_in_offset_inv = T_marker_in_env
        则: subtract(offset_inv, marker) = body_in_env? 不对...
        
      正确方法:
        直接用 combine + 逆变换
    """
    from isaaclab.sim.utils import get_current_stage, resolve_prim_pose
    from isaaclab.utils.math import combine_frame_transforms, quat_inv, quat_apply
    import torch
    
    # 1. 计算Xform相对于Body刚体的固定偏移(只需计算一次)
    stage = get_current_stage()
    
    # 构建完整路径
    # 从 set_Xform_prim_path 中提取父路径作为body路径
    # 例如: "/World/envs/env_0/Robot/fingertip/index_tip_head" -> "/World/envs/env_0/Robot/fingertip"
    body_prim_path = "/".join(set_Xform_prim_path.split("/")[:-1])
    xform_prim_path = set_Xform_prim_path
    
    # 获取USD Prim
    body_prim = stage.GetPrimAtPath(body_prim_path)
    xform_prim = stage.GetPrimAtPath(xform_prim_path)
    
    if not body_prim.IsValid():
        raise ValueError(f"无效的Body Prim路径: {body_prim_path}")
    if not xform_prim.IsValid():
        raise ValueError(f"无效的Xform Prim路径: {xform_prim_path}")
    
    # 计算Marker相对于Body的偏移 (静态计算，不受仿真影响)
    offset_pos_tuple, offset_quat_tuple = resolve_prim_pose(xform_prim, ref_prim=body_prim)
    
    # 转换为torch tensor
    offset_pos = torch.tensor(offset_pos_tuple, device=robot.device, dtype=torch.float32)
    offset_quat = torch.tensor(offset_quat_tuple, device=robot.device, dtype=torch.float32)
    
    print(f"[INFO] Marker到Body的固定偏移:")
    print(f"       body_name: {body_name}")
    print(f"       marker_path: {xform_prim_path}")
    print(f"       offset_pos: {offset_pos.cpu().numpy()}")
    print(f"       offset_quat (wxyz): {offset_quat.cpu().numpy()}")
    
    # 2. 计算偏移的逆变换
    # T^{-1} = [R^T, -R^T * t; 0, 1]
    # 对于四元数表示: (t, q)^{-1} = (-R^T * t, q^*)
    offset_quat_inv = quat_inv(offset_quat)
    offset_pos_inv = quat_apply(offset_quat_inv.unsqueeze(0), -offset_pos.unsqueeze(0))[0]
    
    print(f"       offset_pos_inv: {offset_pos_inv.cpu().numpy()}")
    print(f"       offset_quat_inv (wxyz): {offset_quat_inv.cpu().numpy()}")
    
    # 3. 将所有目标位姿从Marker环境坐标转换到Body环境坐标
    # T_body_in_env = T_marker_in_env × T_marker_in_body^{-1}
    converted_goals = []
    
    for goal in set_goal:
        # 解析Marker在环境中的目标位姿 [x,y,z,qw,qx,qy,qz]
        goal_marker_pos = torch.tensor(goal[:3], device=robot.device, dtype=torch.float32)
        goal_marker_quat = torch.tensor(goal[3:], device=robot.device, dtype=torch.float32)
        
        # 应用逆偏移变换: body = marker × offset^{-1}
        goal_body_pos, goal_body_quat = combine_frame_transforms(
            goal_marker_pos.unsqueeze(0), goal_marker_quat.unsqueeze(0),  # Marker在env中
            offset_pos_inv.unsqueeze(0), offset_quat_inv.unsqueeze(0)     # 偏移的逆
        )
        
        # 转换为列表格式 [x,y,z,qw,qx,qy,qz]
        goal_body = goal_body_pos[0].tolist() + goal_body_quat[0].tolist()
        converted_goals.append(goal_body)
    
    return converted_goals

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="pinv")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for Leaphand fingerti/index_tip_head, 这个是在环境自身坐标系下的目标位置
    # ee_goals_set = [
    #     [-0.0364, -0.1175, 0.6144, 0.393, -0.716, 0.240, -0.525],  # 对应 a_0~a_3：0.35, 0.42, 0.92, 0.86
    #     [-0.0514, -0.1475, 0.6133, 0.389, -0.544, 0.4226, -0.610],  # -0.11, 0.44, 0.36, 1.11
    #     [-0.0385, -0.0653, 0.5743, -0.298, -0.543, 0.006, -0.785],  # -0.56, 0.89, 1.16, 1.53
    # ]
    ee_goals = [
        [-0.0472, -0.1499, 0.5773, 0.394, -0.715, 0.241, -0.524],  #  对应 a_0~a_3：0.35, 0.42, 0.92, 0.86
        [-0.066, -0.1623, 0.5670, 0.388, -0.542, 0.427, -0.611],  #  -0.11, 0.44, 0.36, 1.11
        [-0.0741, -0.0998, 0.5832, -0.3, -0.542, 0.006, -0.785]  # -0.56, 0.89, 1.16, 1.53
    ]

    # # 从设置的指尖坐标转换为fingertip刚体相对于环境根坐标系下的坐标
    # ee_goals = _convert_Xform_to_body_frame(robot, set_goal=ee_goals_set, body_name="fingertip", 
    #                                         set_Xform_prim_path="/World/envs/env_0/Robot/fingertip/index_tip_head")

    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["a_0", "a_1", "a_2", "a_3"], 
                                      body_names=["fingertip"], preserve_order=True)
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 150 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            root_pose_w = robot.data.root_pose_w
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = LeapHandSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
