# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand手内旋转任务环境配置类"""

# 导入LeapHand机器人模型配置
from leaphand.robots.leaphand import LEAPHAND_CONFIG
from pathlib import Path

# 导入相关模块
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

object_usd_path = "/home/hac/LEAP_Hand_Sim/assets/cube/cube.usd"  # 单独的物体USD文件 - 注释掉URDF转换的cube

@configclass
class LeaphandEnvCfg(DirectRLEnvCfg):
    """LeapHand手内旋转任务环境配置类,继承自DirectRLEnvCfg"""

    # 环境基本参数配置
    decimation = 8  # 每隔8个物理仿真步骤执行一次策略 (更接近真实硬件15Hz)
    episode_length_s = 15.0  # 每个episode的时长(秒) - 手内旋转任务相对较短
    action_space = 16  # LeapHand有16个可控关节
    # 观测空间维度计算: 物体位置(3) + 物体旋转(4) + 目标旋转(4) + 手指关节位置(16) + 手指关节速度(16) + 指尖位置(4*3=12) + 物体相对位置(3) + 物体角速度(3) = 61
    observation_space = 61
    state_space = 0  # 状态空间维度(0表示不使用状态空间)
    asymmetric_obs = False  # 是否使用非对称观测
    obs_type = "full"  # 观测类型:"full"表示完整观测
    
    # 仿真器配置
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,  # 仿真时间步长
        render_interval=decimation,  # 渲染间隔
        physics_material=RigidBodyMaterialCfg(  # 物理材质参数
            static_friction=0.5,  # 静摩擦系数
            dynamic_friction=0.5,  # 动摩擦系数
        ),
        physx=PhysxCfg(  # PhysX物理引擎配置
            bounce_threshold_velocity=0.2,  # 反弹阈值速度
        ),
    )
    
    # 机器人配置 - 使用单独的手部USD文件
    robot_cfg = LEAPHAND_CONFIG.replace(prim_path="/World/envs/env_.*/hand")  # 加载LeapHand机器人模型，设置专门的prim路径

    # 可控关节名称列表
    actuated_joint_names = [
        # 食指 index finger
        "a_1", # 连接plam_lower和mcp_joint
        "a_0", # 连接mcp_joint和pip
        "a_2", # 连接pip和dip
        "a_3", # 连接dip和fingertip

        # 中指 middle finger
        "a_5", # 连接plam_lower和mcp_joint2
        "a_4", # 连接mcp_joint2和pip2
        "a_6", # 连接pip2和dip2
        "a_7", # 连接dip2和fingertip2

        # 无名指 ring finger
        "a_9", # 连接plam_lower和mcp_joint3
        "a_8", # 连接mcp_joint3和pip3
        "a_10", # 连接pip3和dip3
        "a_11", # 连接dip3和fingertip3
        
        # 拇指 thumb
        "a_12", # 连接plam_lower和pip_4
        "a_13", # 连接pip4和thumb_pip
        "a_14", # 连接thumb_pip和thumb_dip
        "a_15", # 连接thumb_dip和thumb_fingertip
    ]
    
    # 指尖刚体名称列表
    fingertip_body_names = [
        "fingertip", # 食指指尖
        "fingertip_2", # 中指指尖
        "fingertip_3", # 无名指指尖
        "thumb_fingertip", # 拇指指尖
    ]

    # 目标物体配置 - 使用与实际物体相同的模型，便于可视化目标旋转
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",  # 目标物体路径
        markers={
            "goal": sim_utils.CuboidCfg( 
                size=(0.04, 0.04, 0.04),  # 与实际物体相同的大小
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),  # 绿色目标
            )
        },
    )
    
    # 物体配置 - 使用单独的物体USD文件
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object", # 物体路径 - 使用单独的object路径
        spawn=sim_utils.UsdFileCfg(
            usd_path=object_usd_path,  # 使用单独的物体USD文件
            scale=(0.8, 0.8, 0.8),  # 缩放比例 - 调整为合适的大小
            rigid_props=sim_utils.RigidBodyPropertiesCfg(  # 刚体属性配置
                kinematic_enabled=False,  # 是否为刚体
                disable_gravity=False,  # 是否禁用重力
                enable_gyroscopic_forces=True,  # 是否启用陀螺力
                solver_position_iteration_count=8,  # 位置求解迭代次数
                solver_velocity_iteration_count=0,  # 速度求解迭代次数
                sleep_threshold=0.005,  # 睡眠阈值
                stabilization_threshold=0.0025,  # 稳定化阈值
                max_depenetration_velocity=1000.0,  # 最大分离速度
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1, density=400.0),  # 质量属性 - 使用与Allegro Hand相似的密度
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos=(-0.043, 0.042, 0.183),  # 基于USDA文件中的位置，相对于手部位置
            pos=(-0.043, 0.042, 0.183),  # 基于USDA文件中的位置，相对于手部位置
            rot=(1.0, 0.0, 0.0, 0.0),  # 初始旋转
            lin_vel=(0.0, 0.0, 0.0),  # 初始线速度
            ang_vel=(0.0, 0.0, 0.0),  # 初始角速度
        ),
    )

    # 场景配置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=100,  # 并行环境数量
        env_spacing=0.75,  # 环境间距
        replicate_physics=True  # 是否复制物理引擎
    )

    # 重置参数配置
    reset_position_noise = 0.005  # 重置时位置噪声范围 - 手内旋转需要更精确的位置
    reset_dof_pos_noise = 0.1  # 重置时关节位置噪声范围 - 减小噪声避免物体掉落
    reset_dof_vel_noise = 0.0  # 重置时关节速度噪声范围

    # 手内旋转任务特定参数
    rotation_reward_scale = 10.0  # 旋转奖励系数 - 主要奖励
    rotation_tolerance = 0.1  # 旋转成功容差 (弧度)
    grasp_reward_scale = 5.0  # 抓取奖励系数 - 鼓励保持抓取
    stability_reward_scale = 2.0  # 稳定性奖励系数 - 鼓励平稳旋转

    # 奖励函数相关参数
    dist_reward_scale = -2.0  # 距离奖励系数 - 降低权重，主要关注旋转
    rot_reward_scale = 1.0  # 旋转奖励系数
    rot_eps = 0.05  # 旋转误差阈值 - 更严格的阈值
    action_penalty_scale = -0.001  # 动作惩罚系数 - 增加惩罚鼓励平滑动作
    reach_goal_bonus = 100  # 达到目标奖励 - 降低以平衡奖励
    fall_penalty = -50  # 跌落惩罚 - 增加惩罚
    fall_dist = 0.15  # 跌落距离阈值 - 更严格的阈值 - 更大的值允许更自由的跌落
    vel_obs_scale = 0.2  # 速度观测缩放系数
    success_tolerance = 0.15  # 成功容差 - 更严格
    max_consecutive_success = 5  # 最大连续成功次数 - 增加挑战
    av_factor = 0.1  # 平均因子
    act_moving_average = 0.8  # 动作移动平均 - 增加平滑性
    force_torque_obs_scale = 10.0  # 力矩观测缩放系数

    # 目标旋转参数
    target_rotation_range = 3.14159  # 目标旋转范围 (弧度) - 最大π弧度旋转
    rotation_axis_noise = 0.1  # 旋转轴噪声
