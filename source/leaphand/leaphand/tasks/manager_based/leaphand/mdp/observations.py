# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的观测函数"""

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# rotation_axis观测函数已移除，现在使用Command管理器的generated_commands

def object_pose_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """获取物体世界坐标系下的位姿
    
    Args:
        env: 环境实例
        object_cfg: 物体资产配置
        
    Returns:
        物体位姿 (num_envs, 7) - 位置(3) + 四元数(4)
    """
    # 获取物体资产
    object_asset: RigidObject = env.scene[object_cfg.name]
    
    # 获取位置和旋转
    pos = object_asset.data.root_pos_w
    rot = object_asset.data.root_quat_w  # 四元数
    
    # 拼接位姿
    return torch.cat([pos, rot], dim=-1)


def object_velocity_w(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """获取物体世界坐标系下的速度
    
    Args:
        env: 环境实例
        object_cfg: 物体资产配置
        
    Returns:
        物体速度 (num_envs, 6) - 线速度(3) + 角速度(3)
    """
    # 获取物体资产
    object_asset: RigidObject = env.scene[object_cfg.name]
    
    # 获取线速度和角速度
    lin_vel = object_asset.data.root_lin_vel_w
    ang_vel = object_asset.data.root_ang_vel_w
    
    # 拼接速度
    return torch.cat([lin_vel, ang_vel], dim=-1)


###
# 参考LEAP_Hand_Sim观测项
###

def joint_pos_targets(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """获取关节目标位置
    
    Args:
        env: 环境实例
        asset_cfg: 机器人资产配置
        
    Returns:
        关节目标位置 (num_envs, num_joints)
    
    NOTE:
        关节目标位置是PD控制器的目标位置，不是关节的实际位置
    """
    # 获取机器人资产
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取关节目标位置
    return asset.data.joint_pos_target[:, asset_cfg.joint_ids]


def work_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """获取关节做功惩罚项

    Args:
        env: 环境实例
        asset_cfg: 机器人资产配置

    Returns:
        做功惩罚 (num_envs,)

    NOTE:
        与 LEAP_Hand_Sim 中保持一致：先计算每个环境的关节功率和（关节力矩 * 关节角速度）的和，
        然后对该和取平方作为惩罚，形式为 ((τ * ω).sum(-1)) ** 2。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 关节力矩与关节角速度（均按 asset_cfg.joint_ids 筛选）
    # IsaacLab ArticulationData 使用 applied_torque 作为关节力矩缓冲
    joint_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # 每个环境上各关节τ * ω之和，然后平方（与 leap_hand_rot.py 中的实现一致）
    work = torch.sum(joint_torque * joint_vel, dim=-1)
    work_pen = work ** 2

    return work_pen


def phase(env: ManagerBasedRLEnv, period: float = 2.0) -> torch.Tensor:
    """获取旋转周期的相位编码
    
    Args:
        env: 环境实例
        period: 旋转周期(秒)，默认2秒
        
    Returns:
        相位编码 (num_envs, 2) - [sin(θ), cos(θ)]，其中θ是当前相位角
        
    NOTE:
        - 使用sin-cos编码可以避免相位角度的不连续性
        - 相位角 = 2π * (当前时间 / 周期)
        - 当前时间 = 步数 * 仿真时间步长 * 抽样倍率
    """
    # 计算当前时间和相位角
    current_time = env.common_step_counter * env.cfg.sim.dt * env.cfg.decimation
    # 将标量相位角转换为张量，并放到与环境相同的 device 上
    device = getattr(env, "device", None)
    if device is None and hasattr(env, "sim") and hasattr(env.sim, "device"):
        device = env.sim.device
    phase_scalar = (2.0 * math.pi * current_time) / period
    phase_angle = torch.as_tensor(phase_scalar, device=device)

    # 生成 sin-cos 编码，并扩展到 (num_envs, 2)
    phase_vec = torch.stack([torch.sin(phase_angle), torch.cos(phase_angle)], dim=-1)
    return phase_vec.expand(env.num_envs, -1)
