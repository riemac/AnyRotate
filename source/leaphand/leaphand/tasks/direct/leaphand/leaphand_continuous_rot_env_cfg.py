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
    episode_length_s = 30.0  # 增加episode长度，让手有更多时间进行连续旋转
    
    # 观测空间维度 - 连续旋转任务的观测空间
    # 完整观测: 物体位置(3) + 物体旋转(4) + 手指关节位置(16) + 手指关节速度(16) + 指尖位置(12) + 物体相对位置(3) + 物体角速度(3) + 旋转轴(3) = 60
    # 简化观测: 物体旋转(4) + 手指关节位置(16) + 旋转轴(3) = 23
    observation_space = 60  # 使用完整观测空间
    
    # 连续旋转任务特定参数
    rotation_velocity_reward_scale = 15.0  # 旋转速度奖励系数 - 主要奖励
    rotation_axis_mode = "mixed"  # 旋转轴模式: "random", "z_axis", "mixed"
    rotation_axis_noise = 0.05  # 旋转轴噪声 - 减少噪声以获得更稳定的旋转
    change_rotation_axis_interval = 0  # 更换旋转轴的间隔（步数），0表示不更换
    
    # 调整现有奖励参数以适应连续旋转
    grasp_reward_scale = 8.0  # 增加抓取奖励，确保物体不掉落
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
