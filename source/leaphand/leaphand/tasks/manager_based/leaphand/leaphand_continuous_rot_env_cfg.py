# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务环境配置 - ManagerBasedRLEnv架构"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


import isaaclab.envs.mdp as mdp
from leaphand.robots.leap import LEAP_HAND_CFG
from . import mdp as leaphand_mdp

# 使用Isaac Lab内置的cube资产
object_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"


##
# Scene definition
##
@configclass
class LeaphandContinuousRotSceneCfg(InteractiveSceneCfg):
    """LeapHand连续旋转任务场景配置"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )

    # 机器人
    robot: ArticulationCfg = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 物体
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=object_usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(0.8, 0.8, 0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.1, 0.56),
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # 光照
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class EventCfg:
    """域随机化配置 - 集成官方LeapHand的RL技巧"""

    # -- 机器人域随机化
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,  # 每720步重新随机化
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (3.0, 3.0),
            "damping_distribution_params": (0.1, 0.1),
            "distribution": "uniform",
        },
    )

    # -- 物体域随机化
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- 重置事件
    reset_scene_to_default = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset"
    )


@configclass
class ActionsCfg:
    """动作配置 - 手部关节控制"""

    # 手部关节位置控制
    hand_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],  # 所有手部关节
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """观测配置 - 支持非对称Actor-Critic"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor策略观测 - 真实世界可获取的信息"""

        # 手部关节状态
        hand_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        hand_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})

        # 指尖位置（相对于手掌）
        fingertip_pos = ObsTerm(
            func=leaphand_mdp.relative_fingertip_positions,
            params={"robot_cfg": SceneEntityCfg("robot")}
        )

        # 上一步动作
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "hand_joint_pos"})

        # 当前任务的目标旋转轴
        rotation_axis = ObsTerm(func=leaphand_mdp.rotation_axis)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic价值函数观测 - 包含特权信息"""

        # 继承Policy的所有观测
        hand_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        hand_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        fingertip_pos = ObsTerm(
            func=leaphand_mdp.relative_fingertip_positions,
            params={"robot_cfg": SceneEntityCfg("robot")}
        )
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "hand_joint_pos"})
        rotation_axis = ObsTerm(func=leaphand_mdp.rotation_axis)

        # Critic特权信息（仅在仿真中可获取）
        object_pose = ObsTerm(func=leaphand_mdp.object_pose_w, params={"object_cfg": SceneEntityCfg("object")})
        object_velocity = ObsTerm(func=leaphand_mdp.object_velocity_w, params={"object_cfg": SceneEntityCfg("object")})
        hand_joint_torque = ObsTerm(func=mdp.joint_effort, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # 观测组配置
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """奖励配置 - 连续旋转任务奖励机制"""

    # 主要奖励：旋转速度奖励
    rotation_velocity = RewTerm(
        func=leaphand_mdp.rotation_velocity_reward,
        weight=15.0,
        params={"asset_cfg": SceneEntityCfg("object")},
    )

    # 抓取奖励：保持物体在手中
    grasp_reward = RewTerm(
        func=leaphand_mdp.grasp_reward,
        weight=8.0,
        params={"object_cfg": SceneEntityCfg("object")},
    )

    # 稳定性奖励：减少不必要的震荡
    stability_reward = RewTerm(
        func=leaphand_mdp.stability_reward,
        weight=3.0,
        params={"object_cfg": SceneEntityCfg("object")},
    )

    # 动作惩罚：鼓励平滑动作
    action_penalty = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.0005,
    )


@configclass
class TerminationsCfg:
    """终止条件配置"""

    # 物体掉落终止
    object_falling = DoneTerm(
        func=leaphand_mdp.object_falling_termination,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "fall_dist": 0.12,
            "target_pos_offset": (0.0, -0.1, 0.56),
        },
    )

    # 超时终止
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class LeaphandContinuousRotEnvCfg(ManagerBasedRLEnvCfg):
    """LeapHand连续旋转任务环境配置 - ManagerBasedRLEnv架构"""

    # 场景配置
    scene: LeaphandContinuousRotSceneCfg = LeaphandContinuousRotSceneCfg(
        num_envs=4096,
        env_spacing=0.75,
        replicate_physics=False
    )

    # 环境基本参数
    decimation = 4  # 与官方保持一致
    episode_length_s = 120.0  # 更长的episode以支持连续旋转

    # 仿真配置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**18,
            gpu_max_rigid_patch_count=2**18,
        ),
    )

    # Manager配置
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # 连续旋转任务特定参数
    rotation_axis_mode: str = "z_axis"  # "random", "z_axis", "mixed"
    rotation_axis_noise: float = 0.05
    change_rotation_axis_interval: int = 0  # 0表示不更换
    
    # 课程学习参数
    curriculum_stages: list = ["z_axis", "x_axis", "y_axis", "mixed"]
    curriculum_success_threshold: float = 0.8
    curriculum_steps_per_stage: int = 1000000
    
    # 物理参数
    fall_penalty: float = -100.0
    fall_dist: float = 0.12
    
    # 动作平滑参数
    act_moving_average: float = 0.85
    
    # 历史窗口配置
    history_length: int = 3
    
    # ADR配置（自适应域随机化）
    enable_adr: bool = True
    starting_adr_increments: int = 0
    min_rot_adr_coeff: float = 0.15
    min_steps_for_dr_change: int = 960  # 240 * 4
