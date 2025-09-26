# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务环境配置 - ManagerBasedRLEnv架构
- 该配置类的奖励项将参考LEAP_Hand_Sim中的奖励项
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab.managers import RecorderManagerBaseCfg as DefaultEmptyRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.envs.common import ViewerCfg
from isaaclab.devices.openxr import XrCfg

import isaaclab.envs.mdp as mdp
from leaphand.robots.leap import LEAP_HAND_CFG
from leaphand.tasks.manager_based.leaphand.mdp import observations_privileged as priv_obs
from leaphand.tasks.manager_based.leaphand.mdp.rewards import pose_diff_penalty
from . import mdp as leaphand_mdp
from .mdp.commands import RotationAxisCommandCfg

# 全局超参数(来源于rl_games_ppo_cfg.yaml)
# num_envs = 100
# horizon_length = 240

# 使用Isaac Lab内置的cube资产
object_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"

# Scene definition

@configclass
class InHandSceneCfg(InteractiveSceneCfg):
    """LeapHand连续旋转任务场景配置"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
    )

    # 机器人 - 修改初始关节位置让手指有适当弯曲（与官方一致）
    robot: ArticulationCfg = LEAP_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(0.5, 0.5, -0.5, 0.5),
            joint_pos={
                "a_1": 0.000, "a_12": 0.500, "a_5": 0.000, "a_9": 0.000,
                "a_0": -0.750, "a_13": 1.300, "a_4": 0.000, "a_8": 0.750,
                "a_2": 1.750, "a_14": 1.500, "a_6": 1.750, "a_10": 1.750,
                "a_3": 0.000, "a_15": 1.000, "a_7": 0.000, "a_11": 0.000,
            },
            joint_vel={"a_.*": 0.0},
        )
    )

    # 物体配置 - 用于连续旋转任务的立方体
    object: RigidObjectCfg = RigidObjectCfg(
        # USD场景路径：每个环境实例都有独立的物体
        prim_path="{ENV_REGEX_NS}/object",
        
        # 生成配置：从USD文件加载立方体并设置物理属性
        spawn=sim_utils.UsdFileCfg(
            # USD资产路径：使用Isaac Nucleus中的可变形立方体
            usd_path=object_usd_path,
            
            # 刚体物理属性配置
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 动力学模式：False = 动力学刚体，受物理力影响（重力、碰撞等）
                # True = 运动学刚体，只能通过代码直接控制位置，不受物理力影响
                kinematic_enabled=False,
                
                # 重力开关：False = 物体受重力影响，会自然掉落
                # True = 禁用重力，物体悬浮在空中
                disable_gravity=False,
                
                # 陀螺力：True = 启用陀螺效应，旋转物体时会产生陀螺力矩
                # 这对旋转任务很重要，让物体的旋转行为更真实
                enable_gyroscopic_forces=True,
                
                # 位置求解器迭代次数：控制碰撞检测和位置校正的精度
                # 值越高精度越高但性能开销越大，8是手部操作任务的推荐值
                solver_position_iteration_count=8,
                
                # 速度求解器迭代次数：控制速度约束的求解精度
                # 0表示使用PhysX默认值，通常用于提高计算效率
                solver_velocity_iteration_count=0,
                
                # 休眠阈值：当物体速度低于此值时进入休眠状态以节省计算
                # 0.005 m/s是合理的阈值，避免微小振动浪费计算资源
                sleep_threshold=0.005,
                
                # 稳定化阈值：用于防止小的穿透和抖动
                # 较小的值让物体接触更稳定，特别重要对于精细操作
                stabilization_threshold=0.0025,
                
                # 最大去穿透速度：防止物体在碰撞时以过高速度分离
                # 1000.0是一个很高的值，允许快速的碰撞响应
                max_depenetration_velocity=1000.0,
            ),
            
            # 质量属性：通过密度自动计算物体质量
            # 400.0 kg/m³ 相当于轻木材的密度，适合手部操作
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            
            # 缩放系数：(1.2, 1.2, 1.2) 表示在XYZ三个方向都放大1.2倍
            # 让立方体稍大一些，更容易被手抓取和操作
            scale=(1.2, 1.2, 1.2),
        ),
        
        # 初始状态配置：定义物体在环境重置时的初始位置和姿态
        init_state=RigidObjectCfg.InitialStateCfg(
            # 初始位置：(x=0.0, y=-0.1, z=0.56)
            # z=0.56是在LeapHand手部上方的合适高度
            # y=-0.1稍微偏离中心，给抓取提供更好的角度
            pos=(0.0, -0.1, 0.56),  # root_pos_w -0.05比-0.1稍微更偏近手掌中心，相对更容易抓取
            
            # 初始旋转：(w=1.0, x=0.0, y=0.0, z=0.0)
            # 这是单位四元数，表示无旋转（立方体的标准朝向）
            rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # 光照
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
 
 
@configclass
class CommandsCfg:
    """Commands specifications for the MDP."""
    rotation_axis = RotationAxisCommandCfg(
        rotation_axis_mode="z_axis",  # 默认Z轴旋转
        resampling_time_range=(1e6, 1e6),  # 不基于时间重采样
        change_rotation_axis_interval=0,  # 不自动更换旋转轴
        rotation_axis_noise=0.05,  # 轻微噪声
        debug_vis=True,  # 启用旋转轴可视化
    )
 
 
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    hand_joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],  # 所有手部关节
        scale=1 / 10,  # 增量缩放因子：控制每步的最大位置变化量
        use_zero_offset=True,  # 使用零偏移（相对控制的标准设置）
    )


@configclass
class ObservationsCfg:
    """观测配置 - 支持非对称Actor-Critic"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor策略观测 - 真实世界可获取的信息"""
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        joint_pos_targets = ObsTerm(
            func=leaphand_mdp.joint_pos_targets,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        rotation_axis = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "rotation_axis"},
            history_length=0,  # 明确禁用历史，始终使用当前值
        )

        phase = ObsTerm(
            func=leaphand_mdp.phase,
            params={"period": 2.0},
        )

    @configclass
    class CriticCfg(ObsGroup):
        """Critic价值函数观测 - 包含特权信息"""
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        joint_pos_targets = ObsTerm(
            func=leaphand_mdp.joint_pos_targets,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        rotation_axis = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "rotation_axis"},
            history_length=0,  # 明确禁用历史，始终使用当前值
        )

        phase = ObsTerm(
            func=leaphand_mdp.phase,
            params={"period": 2.0},
        )

        object_pos_w = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("object")},
        )

        object_rot_w = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("object")},
        )

        # 域随机化的特权观测（仅critic可见）：
        # 以“当前/默认”的缩放比为核心特征，并对每组关节做均值/标准差统计，保证输入维度稳定。
        robot_stiffness_stats = ObsTerm(
            func=priv_obs.robot_joint_stiffness_stats,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_damping_stats = ObsTerm(
            func=priv_obs.robot_joint_damping_stats,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_armature_stats = ObsTerm(
            func=priv_obs.robot_joint_armature_stats,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_joint_friction_stats = ObsTerm(
            func=priv_obs.robot_joint_friction_stats,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        object_mass_scale = ObsTerm(
            func=priv_obs.object_total_mass_scale,
            params={"object_cfg": SceneEntityCfg("object")},
        )
        object_material = ObsTerm(
            func=priv_obs.object_material_friction_restitution,
            params={"object_cfg": SceneEntityCfg("object")},
        )

        object_scale_ratio = ObsTerm(
            func=priv_obs.object_scale_ratio,
            params={"object_cfg": SceneEntityCfg("object")},
        )
        object_com_offset = ObsTerm(
            func=priv_obs.object_com_offset,
            params={"object_cfg": SceneEntityCfg("object")},
        )

    # 观测组配置
    policy: PolicyCfg = PolicyCfg(history_length=3)
    critic: CriticCfg = CriticCfg(history_length=3)


@configclass
class RewardsCfg:
    """奖励配置 - 连续旋转任务奖励机制"""
    # rotate_visiualizer = RewTerm(  # 仅用于可视化实际旋转轴
    #     func=leaphand_mdp.rotation_velocity,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "visualize_actual_axis": True,  # 启用实际旋转轴可视化
    #         "target_angular_speed": 1,  # 目标角速度 (rad/s)
    #         "positive_decay": 3.0,  # 正向奖励的指数衰减因子
    #         "negative_penalty_weight": 0.5,  # 负向惩罚权重
    #     },
    # )

    rotate_reward = RewTerm(
        func=leaphand_mdp.rotate_angvel_clipped,
        weight=1.25,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "clip_min": -0.25,
            "clip_max": 0.25,
        },
    )

    object_linvel_penalty = RewTerm(
        func=leaphand_mdp.object_linvel_l1_penalty,
        weight=-0.3,
        params={"object_cfg": SceneEntityCfg("object")},
    )

    pose_diff_penalty = RewTerm( # TODO：该项惩罚有些过高，后期应调整
        func=leaphand_mdp.pose_diff_penalty,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    hand_torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    hand_work_penalty = RewTerm(
        func=leaphand_mdp.work_penalty,
        weight=-0.01,  # 🔥 修复：从-1.0降低到-0.01，减少对动作的过度抑制
                       # 原来的-1.0权重导致策略学会使用极小的动作来避免功率惩罚
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    object_fall_penalty = RewTerm(
        func=leaphand_mdp.object_fall_penalty, # TODO: 该项有些过低
        weight=-10,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "z_threshold": 0.10,
        },
    )


@configclass
class TerminationsCfg:
    """终止条件配置"""

    # 物体掉落终止
    object_falling = DoneTerm(
        # 使用 z 轴高度差判定的终止函数，和 object_fall_penalty 的 z_threshold 逻辑保持一致
        func=leaphand_mdp.object_falling_z_termination,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "z_threshold": 0.10,
        },
    )

    # 超时终止
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg: #
    """域随机化配置 - 集成官方LeapHand的RL技巧"""
    reset_scene_to_default = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset"
    )

    randomized_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.25, 1.2),
            "operation": "scale",  # NOTE: 这个是对质量乘法缩放，缩放系数为上面那个
            "distribution": "uniform",
        },
    )

    randomized_object_com = EventTerm(
        func=leaphand_mdp.randomize_rigid_object_com,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )

    randomized_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": (0.9, 1.1),
        },
    )

    randomized_object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.2, 0.8),  # 塑料、橡胶一般这么多
            "dynamic_friction_range": (0.15, 0.5),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 250,
            "make_consistent": True,  # 确保 dynamic_friction <= static_friction
        },
    )

    randomized_hand_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="a_.*"),
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.6, 1.5),
            "lower_limit_distribution_params": (0.975, 1.025),  # NOTE: 这里是关节限位范围，不是关节阻尼范围
            "upper_limit_distribution_params": (0.975, 1.025),  # NOTE: 这里是关节限位范围，不是关节阻尼范围
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomized_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 1.2),
            "distribution": "uniform",
            "operation": "scale",
        },
    )

    randomized_robot_force_disturbance = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                # body_names=["fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"],  # 施加于指尖
                body_names=".*",  # 所有body
            ),
            "force_range": (-0.5, 0.5),  # N 手整体大约0.75kg
            "torque_range": (-0.025, 0.025),  # N*m
        },
    )

    randomized_object_force_disturbance = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "force_range": (-1.0, 1.0),
            "torque_range": (-0.1, 0.1),
        },
    )


@configclass
class CurriculumCfg:
    """课程学习配置 - 提供各种课程学习策略"""
    # pose_diff_penalty_weight = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={
    #         "term_name": "pose_diff_penalty",
    #         "weight": -0.02,
    #         "num_steps": 300
    #     }
    # )
    pass


@configclass
class InHandEnvCfg(ManagerBasedRLEnvCfg):
    """LeapHand连续旋转任务环境配置类 - ManagerBasedRLEnv架构"""
    ui_window_class_type: type | None = ManagerBasedRLEnvWindow
    is_finite_horizon: bool = True
    # 如果replicate_physics=True，场景会对资产进行复制复用，多个环境实例共享底层资产/物理定义。这会导致 USD 层面的变更无法“按 env 维度独立应用”
    # 注意：字段名需为小写的 'scene' 以符合 ManagerBasedRLEnvCfg 的校验
    scene: InteractiveSceneCfg = InHandSceneCfg(num_envs=100, env_spacing=0.75, replicate_physics=False)
    decimation: int = 4
    episode_length_s: float = 60.0
    viewer: ViewerCfg = ViewerCfg()
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        device="cuda:0",
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.5,  # 被randomized_object_friction覆盖
            dynamic_friction=0.5,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**18,
            gpu_max_rigid_patch_count=2**18,
        ),
    )
    seed: int | None = 42  # 确保每次训练都是可重复的
    recorders: object = DefaultEmptyRecorderManagerCfg()
    rerender_on_reset: bool = False
    wait_for_textures: bool = True
    xr: XrCfg | None = None

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        """后初始化钩子 - 可用于自定义验证或调整配置"""
