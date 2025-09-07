# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务环境配置类"""

from isaaclab.utils import configclass

from .leaphand_env_cfg import LeaphandEnvCfg


@configclass
class LeaphandContinuousRotEnvCfg(LeaphandEnvCfg):
    """LeapHand连续旋转任务环境配置类，继承自LeaphandEnvCfg"""

    # 环境基本参数配置 - 连续旋转任务可能需要更长的episode
    decimation = 8  # 增加decimation以更接近真实LeapHand的控制频率 (15Hz)
    episode_length_s = 15.0  # 增加episode长度，让手有更多时间进行连续旋转

    # 启用非对称Actor-Critic观测
    asymmetric_obs = True  # 启用非对称观测模式

    # 配置驱动的观测架构
    observations_cfg = {
        "actor": {
            "history_steps": 2,  # Actor历史窗口长度
            "components": {
                # 真实世界可通过传感器获取的信息
                "dof_pos": True,        # 手部关节角度 (16维)
                "dof_vel": False,        # 手部关节速度 (16维)
                "fingertip_pos": True,  # 指尖位置 (12维: 4指尖 * 3坐标)
                "last_action": True,    # 上一个时间步的动作 (16维)
                "rotation_axis": True,  # 当前任务的目标旋转轴 (3维)
            }
        },
        "critic": {
            "history_steps": 2,  # Critic历史窗口长度
            "components": {
                # 继承Actor的所有组件
                "dof_pos": True,
                "dof_vel": False,
                "fingertip_pos": True,
                "last_action": True,
                "rotation_axis": True,
                # Critic独有的特权信息（仅在仿真中可获取）
                "object_pose": True,        # 物体位姿 (7维: 位置3 + 四元数4)
                "object_vel": True,         # 物体线速度和角速度 (6维: 线速度3 + 角速度3)
                "dof_torque": True,         # 手部关节力矩 (16维)
                "object_properties": True,  # 物体物理属性 (质量等, 1维)
            }
        }
    }

    # 覆盖基类的观测空间配置以匹配实际维度
    # Actor观测空间维度 (不包含历史): dof_pos(16) + dof_vel(16) + fingertip_pos(12) + last_action(16) + rotation_axis(3) = 63
    # 包含历史: 63 * 2 = 126
    observation_space = 94  # Actor观测空间维度 (包含2步历史)

    # Critic状态空间维度 (不包含历史): Actor所有组件(63) + object_pose(7) + object_vel(6) + dof_torque(16) + object_properties(1) = 93
    # 包含历史: 93 * 2 = 186
    state_space = 154  # Critic状态空间维度 (包含2步历史)

    # 连续旋转任务特定参数
    rotation_velocity_reward_scale = 15.0  # 旋转速度奖励系数 - 主要奖励
    rotation_axis_mode = "z_axis"  # 旋转轴模式: "random", "z_axis", "mixed"
    rotation_axis_noise = 0.05  # 旋转轴噪声 - 减少噪声以获得更稳定的旋转
    change_rotation_axis_interval = 0  # 更换旋转轴的间隔（步数），0表示不更换

    # 调整现有奖励参数以适应连续旋转
    grasp_reward_scale = 5.0  # 增加抓取奖励，确保物体不掉落
    stability_reward_scale = 3.0  # 增加稳定性奖励，但不过度限制旋转
    action_penalty_scale = -0.0005  # 减少动作惩罚，允许更大的动作幅度

    # 移除目标相关的参数
    # rotation_reward_scale - 不再需要，用rotation_velocity_reward_scale替代
    # rotation_tolerance - 不再需要，连续旋转没有目标容差
    # reach_goal_bonus - 不再需要，连续旋转没有目标奖励
    # max_consecutive_success - 不再需要，连续旋转没有成功概念
    # target_rotation_range - 不再需要，连续旋转没有目标范围

    # 保持其他参数不变
    fall_penalty = -100  # 增加跌落惩罚，因为连续旋转更容易导致物体掉落
    fall_dist = 0.12  # 稍微减小跌落距离阈值，更严格地检测掉落

    # 重置参数 - 连续旋转可能需要更精确的初始位置
    reset_position_noise = 0.003  # 减少位置噪声
    reset_dof_pos_noise = 0.08  # 减少关节位置噪声
    reset_dof_vel_noise = 0.0  # 保持速度噪声为0

    # 动作平滑参数
    act_moving_average = 0.85  # 增加动作平滑性，有助于连续旋转的稳定性

    # 速度观测缩放 - 连续旋转任务中速度信息更重要
    vel_obs_scale = 0.3  # 增加速度观测的权重

    # 力矩观测缩放
    force_torque_obs_scale = 8.0  # 稍微减少力矩观测的缩放，避免过度敏感

    # 平均因子
    av_factor = 0.05  # 减少平均因子，让奖励更快响应当前表现
