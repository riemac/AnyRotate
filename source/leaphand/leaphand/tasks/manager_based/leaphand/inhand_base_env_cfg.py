# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务环境配置 - ManagerBasedRLEnv架构
- 该配置类的奖项参考LEAP_Hand_Isaac_Lab，尽管任务不同
- 主要增加一个连续旋转目标达成的稀疏奖励项
"""

import math
from shlex import join

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
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.envs.common import ViewerCfg
from isaaclab.devices.openxr import XrCfg

import isaaclab.envs.mdp as mdp
from leaphand.robots.leap import LEAP_HAND_CFG
from leaphand.tasks.manager_based.leaphand.mdp import observations_privileged as priv_obs
from leaphand.tasks.manager_based.leaphand.mdp.rewards import pose_diff_penalty, track_orientation_inv_l2
from . import mdp as leaphand_mdp

# from .mdp.actions import LinearDecayAlphaEMAJointPositionToLimitsActionCfg

# 全局超参数(来源于rl_games_ppo_cfg.yaml)
horizon_length = 32
epochs_num = 5 # 与horizon_length配合以确定数据更新频率

# 使用Isaac Lab内置的cube资产
object_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"

@configclass
class InHandSceneCfg(InteractiveSceneCfg):
    """LeapHand连续旋转任务场景配置"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            usd_path="/home/hac/isaac/isaacsim_assets/Assets/Isaac/5.0/Isaac/Environments/Grid/default_environment.usd"
        ),
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
    goal_pose = leaphand_mdp.ContinuousRotationCommandCfg(
        asset_name="object",
        resampling_time_range=(1e6, 1e6),  # 不基于时间重采样
        init_pos_offset=(0.0, 0.0, 0.0),
        rotation_axis="z",  # 固定Z轴旋转
        delta_angle=math.pi / 8.0,  # 每次旋转22.5度
        make_quat_unique=True,
        update_goal_on_success=True,
        # orientation_success_threshold 将由 __post_init__ 自动计算为 delta_angle/20
    )
 
 
@configclass
class ActionsCfg:
    """动作配置 - 动作平滑"""
    hand_joint_pos = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],  # 所有手部关节
        scale=1 / 10,  # 增量缩放因子：控制每步的最大位置变化量
        use_zero_offset=True,  # 使用零偏移（相对控制的标准设置）
    )
    # hand_joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
    #     asset_name="robot",
    #     joint_names=["a_.*"],  # 所有手部关节
    #     scale=1.0,  # 动作缩放因子（对EMA类型影响不大，因为有rescale_to_limits）
    #     rescale_to_limits=True,  # 将[-1,1]动作自动映射到关节限制
    #     alpha=1/10,  # 平滑系数
    # )

@configclass
class ObservationsCfg:
    """观测配置 - 支持非对称Actor-Critic"""

    @configclass
    class PrivilegedObsCfg(ObsGroup):
        """Actor策略观测 - 包含大量仅仿真可用的特权信息"""
        # -- robot terms
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # -- object terms
        object_pos = ObsTerm(func=mdp.root_pos_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")})
        object_quat = ObsTerm( # IDEA:该项添加噪音可能会破坏归一化约束？
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False}
        )

        # -- command terms
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "goal_pose"})
        goal_quat_diff = ObsTerm(
            func=leaphand_mdp.goal_quat_diff,
            params={"asset_cfg": SceneEntityCfg("object"), "command_name": "goal_pose", "make_quat_unique": True},
        )

        # -- action terms
        last_action = ObsTerm(func=mdp.last_action) # 返回的是 策略输出的规范化后值（通常是 -1 到 1）

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PrivilegedObsCfg):
        """Critic价值函数观测 - 仅包含真实世界可获取的信息"""

    # 观测组配置
    policy: ObsGroup = PrivilegedObsCfg(history_length=2)
    critic: ObsGroup = CriticCfg(history_length=2)


@configclass
class EventCfg: #
    """域随机化配置 - 集成官方LeapHand的RL技巧"""
    
    # -- object
    randomized_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.25, 1.2),
            "operation": "scale", # 这个是对质量乘法缩放，缩放系数为上面那个
            "distribution": "uniform",
        },
    )

    randomized_object_com = EventTerm(
        func=leaphand_mdp.randomize_rigid_object_com,
        mode="startup",
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
        mode="startup",
        params={ # 从 range 里均匀采样
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.2, 1.0), # 塑料、橡胶一般这么多
            "dynamic_friction_range": (0.15, 0.6),
            "restitution_range": (0.0, 0.1), # 不提供的话默认(0,0)
            "num_buckets": 250,
            "make_consistent": True,  # 确保 dynamic_friction <= static_friction
        },
    )

    randomized_object_force_disturbance = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        min_step_count_between_reset=epochs_num*horizon_length,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "force_range": (-1.0, 1.0),
            "torque_range": (-0.1, 0.1),
        },
    )

    # -- robot
    randomized_hand_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="a_.*"),
            "friction_distribution_params": (0.8, 1.2),
            "armature_distribution_params": (0.6, 1.5),
            "lower_limit_distribution_params": (0.975, 1.025),  # 这里是关节限位范围，不是关节阻尼范围
            "upper_limit_distribution_params": (0.975, 1.025),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomized_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1), # 官方默认是3.0
            "damping_distribution_params": (0.8, 1.2), # 官方默认是0.1
            "distribution": "uniform",
            "operation": "scale", # 缩放
        },
    )

    randomized_robot_force_disturbance = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        min_step_count_between_reset=epochs_num*horizon_length,
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

    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05), # 每个机器人连杆的质量都乘以不同随机的缩放系数
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    
    # -- reset
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={ # rpy角是按照Isaaclab中的坐标系惯例的正方向来的
            "pose_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01],
                           "roll": [-0.0, 0.0], "pitch": [-0.0, 0.0], "yaw": [-math.pi, math.pi]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": {".*": [-0.2, 0.2]},
            "velocity_range": {".*": [0.0, 0.0]},
        },
    )


@configclass
class RewardsCfg:
    """奖励配置 - 连续旋转任务奖励机制"""

    # -- task
    track_orientation_inv_l2 = RewTerm(
        func=leaphand_mdp.track_orientation_inv_l2,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("object"), "rot_eps": 0.1, "command_name": "goal_pose"},
    )
    success_bonus = RewTerm(
        func=leaphand_mdp.success_bonus,
        weight=250.0,
        params={"object_cfg": SceneEntityCfg("object"), "command_name": "goal_pose"},
    )

    # -- penalties
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-5)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

@configclass
class TerminationsCfg:
    """终止条件配置"""

    # 物体掉落终止
    object_falling = DoneTerm(func=leaphand_mdp.object_away_from_robot, params={"threshold": 0.3})

    # 超时终止
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """课程学习配置 - 提供各种课程学习策略"""


@configclass
class InHandObjectEnvCfg(ManagerBasedRLEnvCfg):
    """LeapHand连续旋转任务环境配置类 - ManagerBasedRLEnv架构"""
    ui_window_class_type: type | None = ManagerBasedRLEnvWindow
    is_finite_horizon: bool = True
    # 如果replicate_physics=True，场景会对资产进行复制复用，多个环境实例共享底层资产/物理定义。这会导致 USD 层面的变更无法“按 env 维度独立应用”
    # 注意：字段名需为小写的 'scene' 以符合 ManagerBasedRLEnvCfg 的校验
    scene: InteractiveSceneCfg = InHandSceneCfg(num_envs=100, env_spacing=0.75, replicate_physics=False)
    viewer: ViewerCfg = ViewerCfg()
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.5,  # 被 randomized_object_friction 覆盖
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

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # Curriculum settings
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        """后初始化钩子 - 可用于自定义验证或调整配置"""
        # general settings
        self.decimation = 4
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # change viewer settings
        self.viewer.eye = (2.0, 2.0, 2.0)
