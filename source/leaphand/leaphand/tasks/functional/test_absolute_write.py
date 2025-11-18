# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""

直接设置LeapHand指尖的绝对位姿目标，然后逆解关节角度，并通过write_data_to_sim接口写入，看是否能正确驱动手指到达目标位置。

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
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

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

    # articulation
    robot = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for Leaphand fingertip, 这个是在环境自身坐标系下的目标位置
    ee_goals_pose = [
        [-0.0472, -0.1499, 0.5773, 0.394, -0.715, 0.241, -0.524],  #  对应关节角 a_0~a_3：0.35, 0.42, 0.92, 0.86
        [-0.066, -0.1623, 0.5670, 0.388, -0.542, 0.427, -0.611],   #  对应关节角 a_0~a_3：-0.11, 0.44, 0.36, 1.11
        [-0.0741, -0.0998, 0.5832, -0.3, -0.542, 0.006, -0.785]    #  对应关节角 a_0~a_3：-0.56, 0.89, 1.16, 1.53
    ]
    
    # 对应的期望关节角度（从你手动控制获得，用于验证）
    ee_goals_joint = torch.tensor([
        [0.35, 0.42, 0.92, 0.86],
        [-0.11, 0.44, 0.36, 1.11],
        [-0.56, 0.89, 1.16, 1.53]
    ], device=sim.device)

    ee_goals_pose = torch.tensor(ee_goals_pose, device=sim.device)
    current_goal_idx = 0

    # Specify robot-specific parameters（只控制食指前4个关节）
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["a_0", "a_1", "a_2", "a_3"], body_names=["fingertip"], preserve_order=True)
    robot_entity_cfg.resolve(scene)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    hold_time = 0
    
    # 存储当前目标关节角
    current_joint_goal = ee_goals_joint[current_goal_idx].repeat(scene.num_envs, 1)
    
    print(f"\n[INFO] 开始测试直接关节角写入")
    print(f"       目标位姿: {ee_goals_pose[current_goal_idx].cpu().numpy()}")
    print(f"       期望关节角: {current_joint_goal[0].cpu().numpy()}")
    
    # Simulation loop
    while simulation_app.is_running():
        # 每300步切换目标
        if count % 30 == 0 and count > 0:
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals_pose)
            current_joint_goal = ee_goals_joint[current_goal_idx].repeat(scene.num_envs, 1)
            
            print(f"\n[切换目标] 步数: {count}")
            print(f"       目标位姿: {ee_goals_pose[current_goal_idx].cpu().numpy()}")
            print(f"       期望关节角: {current_joint_goal[0].cpu().numpy()}")

        # 直接设置关节角到期望值（一步到位，不做IK迭代）
        joint_pos_full = robot.data.default_joint_pos.clone()
        joint_pos_full[:, robot_entity_cfg.joint_ids] = current_joint_goal  # 这里指定了要写入的关节，与preserve_order的行为有关系
        joint_vel_full = torch.zeros_like(robot.data.default_joint_vel)
        
        # 直接写入关节状态
        robot.write_joint_state_to_sim(joint_pos_full, joint_vel_full)
        scene.write_data_to_sim()
        
        # 执行仿真步
        sim.step()
        count += 1
        scene.update(sim_dt)

        # 获取实际末端位姿用于可视化
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(
            ee_goals_pose[current_goal_idx, 0:3].repeat(scene.num_envs, 1) + scene.env_origins, 
            ee_goals_pose[current_goal_idx, 3:7].repeat(scene.num_envs, 1)
        )
        
        # 每100步打印一次误差
        if count % 100 == 0:
            pos_error = torch.norm(ee_pose_w[0, 0:3] - (ee_goals_pose[current_goal_idx, 0:3] + scene.env_origins[0]), p=2)
            joint_pos_actual = robot.data.joint_pos[0, robot_entity_cfg.joint_ids]
            joint_error = torch.norm(joint_pos_actual - current_joint_goal[0], p=2)
            print(f"[步数 {count:04d}] 位置误差: {pos_error.item():.4f}m, 关节误差: {joint_error.item():.4f}rad")


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
