# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的奖励函数"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_conjugate, quat_mul, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rotation_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """计算旋转速度奖励

    Args:
        env: 环境实例
        asset_cfg: 物体资产配置

    Returns:
        旋转速度奖励 (num_envs,)
    """
    # 获取物体资产
    asset: RigidObject = env.scene[asset_cfg.name]

    # 获取当前物体旋转
    current_object_rot = asset.data.root_quat_w

    # 初始化last_object_rot如果不存在
    if not hasattr(env, 'last_object_rot'):
        env.last_object_rot = torch.zeros((env.num_envs, 4), dtype=torch.float, device=env.device)
        env.last_object_rot[:, 0] = 1.0  # 初始化为单位四元数

    # 计算旋转差异
    quat_diff = quat_mul(current_object_rot, quat_conjugate(env.last_object_rot))
    angle = 2.0 * torch.acos(torch.clamp(torch.abs(quat_diff[:, 0]), max=1.0))

    # 计算旋转轴
    axis = quat_diff[:, 1:4]
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    valid_rotation = axis_norm.squeeze(-1) > 1e-6
    axis = torch.where(valid_rotation.unsqueeze(-1), axis / axis_norm, torch.zeros_like(axis))

    # 获取目标旋转轴
    if not hasattr(env, 'rotation_axis'):
        env.rotation_axis = torch.zeros((env.num_envs, 3), dtype=torch.float, device=env.device)
        env.rotation_axis[:, 2] = 1.0  # 默认Z轴

    # 计算沿指定旋转轴的角速度
    angular_velocity = angle / env.step_dt
    projected_velocity = torch.sum(axis * env.rotation_axis, dim=-1) * angular_velocity

    # 更新上一帧旋转
    env.last_object_rot[:] = current_object_rot.clone()

    # 奖励正向旋转，轻微惩罚反向旋转
    reward = torch.clamp(projected_velocity, min=-0.1)

    return reward


def grasp_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_pos_offset: tuple[float, float, float] = (0.0, -0.1, 0.56)
) -> torch.Tensor:
    """计算抓取奖励 - 鼓励保持物体在手中

    Args:
        env: 环境实例
        object_cfg: 物体资产配置
        target_pos_offset: 目标位置偏移

    Returns:
        抓取奖励 (num_envs,)
    """
    # 获取物体资产
    object_asset: RigidObject = env.scene[object_cfg.name]

    # 计算物体与目标位置的距离
    object_pos = object_asset.data.root_pos_w
    target_pos = torch.tensor(target_pos_offset, device=env.device).expand(env.num_envs, -1)
    object_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    # 指数衰减奖励
    reward = torch.exp(-10.0 * object_dist)

    return reward


def stability_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """计算稳定性奖励 - 减少不必要的震荡

    Args:
        env: 环境实例
        object_cfg: 物体资产配置

    Returns:
        稳定性奖励 (num_envs,)
    """
    # 获取物体资产
    object_asset: RigidObject = env.scene[object_cfg.name]

    # 基于物体线速度的稳定性奖励
    object_lin_vel = object_asset.data.root_lin_vel_w
    lin_vel_penalty = torch.norm(object_lin_vel, p=2, dim=-1)

    # 指数衰减奖励
    reward = torch.exp(-2.0 * lin_vel_penalty)

    return reward


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)
