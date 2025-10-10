# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务环境配置 - ManagerBasedRLEnv架构
"""

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
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


import isaaclab.envs.mdp as mdp
from leaphand.robots.leap import LEAP_HAND_CFG
from . import mdp as leaphand_mdp
from .mdp.commands import RotationAxisCommandCfg

# 全局超参数(来源于rl_games_ppo_cfg.yaml)
num_envs = 100
horizon_length = 240

# 使用Isaac Lab内置的cube资产
object_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"

# Scene definition
@configclass
class LeaphandContinuousRotSceneCfg(InteractiveSceneCfg):
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
            pos=(0.0, -0.1, 0.56), # root_pos_w -0.05比-0.1稍微更偏近手掌中心，相对更容易抓取
            
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
    """命令配置 - 旋转轴目标生成"""

    rotation_axis = RotationAxisCommandCfg(
        rotation_axis_mode="z_axis",  # 默认Z轴旋转
        resampling_time_range=(1e6, 1e6),  # 不基于时间重采样
        change_rotation_axis_interval=0,  # 不自动更换旋转轴
        rotation_axis_noise=0.05,  # 轻微噪声
        debug_vis=True,  # 启用旋转轴可视化
    )


@configclass
class ActionsCfg:
    """动作配置 - 手部关节控制

    提供三种动作控制方案，通过注释切换：
    1. 原始绝对位置控制（会产生抖动）
    2. EMA指数移动平均平滑控制（推荐）
    3. 相对位置增量控制（天然平滑）
    """

    # 方案1：原始绝对位置控制（会产生抖动，仅用于对比）
    # hand_joint_pos = mdp.JointPositionToLimitsActionCfg(
    #     asset_name="robot",
    #     joint_names=["a_.*"],  # 所有手部关节
    #     scale=1.0,  # 动作缩放因子
    #     rescale_to_limits=True,  # 将[-1,1]动作自动映射到关节限制
    # )

    # 方案2：EMA指数移动平均平滑控制（推荐 - 类似官方LeapHand的绝对控制+平滑）
    hand_joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],  # 所有手部关节
        scale=1.0,  # 动作缩放因子
        rescale_to_limits=True,  # 将[-1,1]动作自动映射到关节限制
        alpha=1/24,  # EMA平滑系数：0.1表示当前动作10%权重，历史90%权重（强平滑）
                    # 参考：官方LeapHand使用 1/24≈0.042 (超强平滑) 或者一开始抖动，后面慢慢平滑？
                    # 建议范围：0.05-0.2，值越小越平滑但响应越慢
    )

    # 方案3：相对位置增量控制（天然平滑 - 类似官方LeapHand的相对控制）
    # hand_joint_pos = mdp.RelativeJointPositionActionCfg(
    #     asset_name="robot",
    #     joint_names=["a_.*"],  # 所有手部关节
    #     scale=1/24,  # 增量缩放因子：控制每步的最大位置变化量
    #                  # 参考：官方LeapHand相对模式使用很小的增量
    #                  # 建议范围：0.01-0.1，值越小动作越平滑但学习越慢
    #     use_zero_offset=True,  # 使用零偏移（相对控制的标准设置）
    # )


# @configclass
# class DynamicScaleActionsCfg:
#     """动态缩放动作配置 - 使用自定义的动态缩放ActionTerm

#     这个配置使用自定义的DynamicScaleRelativeJointPositionAction，
#     可以在训练过程中自动调整动作的缩放因子，解决探索与利用的平衡问题。
#     """

#     # 手部关节位置控制 - 带动态缩放因子的相对位置增量控制
#     hand_joint_pos = DynamicScaleRelativeJointPositionActionCfg(
#         asset_name="robot",
#         joint_names=["a_.*"],  # 所有手部关节
#         scale=0.1,  # 初始缩放因子（会被动态调整覆盖）
#         use_zero_offset=True,  # 使用零偏移（相对位置控制）

#         # 动态缩放参数
#         scale_strategy="linear",  # 线性衰减策略
#         alpha_max=0.15,          # 起始缩放因子（较大值，利于前期探索）
#         alpha_min=0.05,          # 终止缩放因子（较小值，利于后期精细控制）
#         start_step=0,            # 立即开始调整
#         end_step=100000,         # 10万步完成调整
#         decay_rate=3.0,          # 指数衰减率（仅用于exponential策略）
#         debug_interval=10000,    # 每10000步输出一次调试信息
#     )


@configclass
class ObservationsCfg:
    """观测配置 - 支持非对称Actor-Critic"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor策略观测 - 真实世界可获取的信息"""

        # 手部关节状态 - 使用标准化到[-1,1]的关节位置，与动作空间对齐
        hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )  # 默认关节速度都是0

        # 上一步动作
        last_action = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "hand_joint_pos"}, # 这里使用的是ActionsCfg中的名称（不同管理器都有独立的命名空间）
        )

        # 当前任务的目标旋转轴 - 使用Command管理器，不使用历史（goal-conditioned策略的条件）
        rotation_axis = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "rotation_axis"},
            history_length=0,  # 明确禁用历史，始终使用当前值
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic价值函数观测 - 包含特权信息"""

        # 继承Policy的所有观测 - 使用标准化到[-1,1]的关节位置
        hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        # 上一步动作 
        last_action = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "hand_joint_pos"}, # 这里使用的是ActionsCfg中的名称（不同管理器都有独立的命名空间）
        )

        # 当前任务的目标旋转轴 - 使用Command管理器，不使用历史
        rotation_axis = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "rotation_axis"},
            history_length=0,  # 明确禁用历史，始终使用当前值
        )

        # Critic特权信息（仅在仿真中可获取）
        object_pose = ObsTerm(
            func=leaphand_mdp.object_pose_w,
            params={"object_cfg": SceneEntityCfg("object")}
        )
        object_velocity = ObsTerm(
            func=leaphand_mdp.object_velocity_w,
            params={"object_cfg": SceneEntityCfg("object")}
        )
        hand_joint_torque = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # 观测组配置
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class RewardsCfg:
    """奖励配置 - 连续旋转任务奖励机制"""

    # 主要奖励：旋转速度奖励 - 目标角速度型
    rotation_velocity = RewTerm(
        func=leaphand_mdp.rotation_velocity,
        weight=40.0,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "visualize_actual_axis": True,  # 启用实际旋转轴可视化
            "target_angular_speed": 1,   # 目标角速度 (rad/s)
            "positive_decay": 3.0,        # 正向奖励的指数衰减因子
            "negative_penalty_weight": 0.5,  # 负向惩罚权重
        },
    )

    # 旋转轴对齐奖励：鼓励实际旋转轴与目标旋转轴对齐
    rotation_axis_alignment_reward = RewTerm(
        func=leaphand_mdp.rotation_axis_alignment_reward,
        weight=20.0,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "theta_tolerance": 10/180*math.pi,  # 角度容忍度 (弧度)
            "decay_factor": 10.0,     # 指数衰减因子
        },
    )

    # 抓取奖励：保持物体在手中
    # grasp_reward = RewTerm(
    #     func=leaphand_mdp.grasp_reward,
    #     weight=4.0,
    #     params={"object_cfg": SceneEntityCfg("object")},
    # )

    # 稳定性惩罚：减少不必要的震荡
    unstable_penalty = RewTerm(
        func=leaphand_mdp.unstable_penalty,
        weight=-5.0, # TODO:这个权重可能需要调大
        params={"object_cfg": SceneEntityCfg("object")},
    )

    # 动作惩罚：鼓励平滑动作
    action_penalty = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1,
    )

    # 姿态偏差惩罚：鼓励保持接近人手的自然姿态
    pose_diff_penalty = RewTerm(
        func=leaphand_mdp.pose_diff_penalty,
        weight=-0.5, # 初始是-1
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 指尖距离惩罚：鼓励机器人靠近物体中心
    fingertip_distance_penalty = RewTerm(
        func=leaphand_mdp.fingertip_distance_penalty,
        weight=-100.0, # 初始是-100
        params={
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    # 扭矩惩罚：鼓励使用较小的关节扭矩
    torque_penalty = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 掉落惩罚：物体掉落时的严重惩罚
    fall_penalty = RewTerm(
        func=leaphand_mdp.fall_penalty,
        weight=-500.0,
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "fall_distance": 0.12
        },
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
class EventCfg:
    """域随机化配置 - 集成官方LeapHand的RL技巧"""

    # -- 场景重置（确保一致的初始状态，正确处理环境偏移）不设置该项的话，物体跌落地面则不会重回手上
    reset_scene_to_default = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset"
    )

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

    # 物体尺寸随机化
    object_scale_size = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",  # 在仿真开始前随机化
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": {"x": (1.0, 1.0), "y": (1.0, 1.0), "z": (1.0, 1.0)},  # 初始无随机化
        },
    )



@configclass
class CurriculumCfg:
    """课程学习配置 - 提供各种课程学习策略"""

    # 动作缩放因子调整课程学习 - 基于RL训练epoch边界调整，解决探索与利用平衡问题
    # action_scale_factor = CurrTerm(
    #     func=leaphand_mdp.modify_action_scale_factor_rl_epoch_based,  # 使用基于RL epoch的函数
    #     params={
    #         "action_term_name": "hand_joint_pos",  # 动作项名称
    #         "alpha_max": 0.15,        # 起始值：较大，利于前期探索
    #         "alpha_min": 0.05,        # 终止值：较小，利于后期精细控制
    #         "start_epoch": 0,         # 立即开始调整
    #         "end_epoch": 10,          # 10个RL epoch完成调整
    #         "epoch_interval": 1,      # 每个epoch都调整
    #         "horizon_length": horizon_length,  # episode长度
    #         "num_envs": num_envs      # 环境数量
    #     }
    # )
    

    # 奖励权重调整课程学习
    # grasp_stability_weight = CurrTerm(
    #     func=leaphand_mdp.modify_grasp_stability_weight,
    #     params={
    #         "term_name": "grasp_stability",
    #         "early_weight": 2.0,
    #         "mid_weight": 1.5,
    #         "late_weight": 1.0,
    #         "mid_step": 500_000,
    #         "late_step": 1_000_000
    #     }
    # )

    rotation_velocity_weight = CurrTerm(
        func=leaphand_mdp.modify_rotation_velocity_weight,
        params={
            "term_name": "rotation_velocity",  # 修复：匹配RewardsCfg中的实际名称
            "early_weight": 10.0,
            "mid_weight": 20.0,
            "late_weight": 40.0,
            "mid_step": 100*horizon_length*num_envs,  # 1,200,000步
            "late_step": 250*horizon_length*num_envs  # 2,880,000步
        }
    )

    rotation_axis_alignment_weight = CurrTerm(
        func=leaphand_mdp.modify_rotation_axis_alignment_weight,
        params={
            "term_name": "rotation_axis_alignment_reward",  # 修复：匹配RewardsCfg中的实际名称
            "early_weight": 10.0,
            "mid_weight": 20.0,
            "late_weight": 40.0,
            "mid_step": 100*horizon_length*num_envs,  # 2,400,000步
            "late_step": 250*horizon_length*num_envs  # 6,000,000步
        }
    )
    
    fingertip_distance_penalty_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "fingertip_distance_penalty",
            "weight": -12.0,
            "num_steps": 100*horizon_length*num_envs
        }
    )
    
    unstable_penalty_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "unstable_penalty",
            "weight": -10.0,
            "num_steps": 200*horizon_length*num_envs
        }
    )

    action_penalty_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "action_penalty",
            "weight": -0.5, # 后期加大动作惩罚
            "num_steps": 200*horizon_length*num_envs
        }
    )

    torque_penalty_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "torque_penalty",
            "weight": -1.2, # 后期加大扭矩惩罚
            "num_steps": 200*horizon_length*num_envs
        }
    )

    # 姿态偏差惩罚权重调整 - 训练初期重，后期变轻
    pose_diff_penalty_weight = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "pose_diff_penalty",
            "weight": -0.2,  # 后期减轻姿态约束
            "num_steps": 100*horizon_length*num_envs
        }
    )


    # 自适应域随机化课程学习
    # object_mass_adr = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "events.object_scale_mass.params.mass_distribution_params",
    #         "modify_fn": leaphand_mdp.object_mass_adr,
    #         "modify_params": {
    #             "enable_step": 600_000,
    #             "max_strength_step": 1_200_000,
    #             "max_variation": 0.5
    #         }
    #     }
    # )

    # friction_adr = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "events.object_physics_material.params.static_friction_range",
    #         "modify_fn": leaphand_mdp.friction_adr,
    #         "modify_params": {
    #             "enable_step": 800_000,
    #             "max_strength_step": 1_500_000,
    #             "max_variation": 0.3
    #         }
    #     }
    # )

    # object_scale_adr = CurrTerm(
    #     func=mdp.modify_env_param,
    #     params={
    #         "address": "events.object_scale_size.params.scale_range",
    #         "modify_fn": leaphand_mdp.object_scale_adr,
    #         "modify_params": {
    #             "enable_step": 1_000_000,
    #             "max_strength_step": 1_800_000,
    #             "max_variation": 0.2
    #         }
    #     }
    # )

    # 旋转轴复杂度课程学习
    # simple_rotation_axis = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "commands.rotation_axis.rotation_axis_mode", # 指向CommandsCfg中的rotation_axis命令项/字段
    #         "modify_fn": leaphand_mdp.simple_rotation_axis,
    #         "modify_params": {
    #             "z_axis_step": 0,
    #             "random_axis_step": 100
    #         }
    #     }
    # )

@configclass
class LeaphandContinuousRotEnvCfg(ManagerBasedRLEnvCfg):
    """LeapHand连续旋转任务环境配置 - ManagerBasedRLEnv架构"""

    # 场景配置
    scene: LeaphandContinuousRotSceneCfg = LeaphandContinuousRotSceneCfg(
        num_envs=100,
        env_spacing=0.75,
        replicate_physics=False
    )

    # 环境基本参数
    decimation = 4  # 与官方保持一致
    episode_length_s = 15.0  # 更长的episode以支持连续旋转

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
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # 课程学习配置 - 启用全部课程学习
    # curriculum: object | None = None  # 禁用课程学习
    curriculum: CurriculumCfg = CurriculumCfg()  # 启用全部课程学习

    # 指尖身体名称列表
    fingertip_body_names = [
        'fingertip', 
        'thumb_fingertip', 
        'fingertip_2', 
        'fingertip_3'
    ]

    # 可驱动关节名称列表
    actuated_joint_names = [
        'a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7',
        'a_8', 'a_9', 'a_10', 'a_11', 'a_12', 'a_13', 'a_14', 'a_15'
    ]

    # 连续旋转任务特定参数
    rotation_axis_mode: str = "z_axis"  # "random", "z_axis", - 初始模式，课程学习会动态调整
    rotation_axis_noise: float = 0.05
    change_rotation_axis_interval: int = 0  # 0表示不更换

    # 物理参数clear
    fall_penalty: float = -100.0
    fall_dist: float = 0.12

    # 历史窗口配置
    history_length: int = 3

    def __post_init__(self):
        """配置后处理：动态设置历史长度和其他参数"""
        # 逐项精确控制历史设置（避免组级配置覆盖项级配置）
        # 为需要历史的观测项设置历史长度，保持rotation_axis和last_action为0

        # Policy观测历史配置
        self.observations.policy.hand_joint_pos.history_length = self.history_length
        self.observations.policy.hand_joint_vel.history_length = self.history_length
        self.observations.policy.last_action.history_length = self.history_length
        # rotation_axis保持history_length=0（已在定义时设置）

        # Critic观测历史配置
        self.observations.critic.hand_joint_pos.history_length = self.history_length
        self.observations.critic.hand_joint_vel.history_length = self.history_length
        self.observations.critic.object_pose.history_length = self.history_length
        self.observations.critic.object_velocity.history_length = self.history_length
        self.observations.critic.hand_joint_torque.history_length = self.history_length
        self.observations.critic.last_action.history_length = self.history_length
        # rotation_axis保持history_length=0（已在定义时设置）

        # 设置历史维度展平
        self.observations.policy.flatten_history_dim = True
        self.observations.critic.flatten_history_dim = True

