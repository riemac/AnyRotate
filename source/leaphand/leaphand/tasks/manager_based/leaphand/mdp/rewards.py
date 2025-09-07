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

    # 获取目标旋转轴 - 从Command管理器获取
    rotation_axis = env.command_manager.get_command("rotation_axis")

    # 计算沿指定旋转轴的角速度
    angular_velocity = angle / env.step_dt
    projected_velocity = torch.sum(axis * rotation_axis, dim=-1) * angular_velocity

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
        target_pos_offset: 目标位置偏移（环境局部坐标系）

    Returns:
        抓取奖励 (num_envs,)
    """
    # 获取物体资产
    object_asset: RigidObject = env.scene[object_cfg.name]

    # 获取物体位置（世界坐标系）
    object_pos_w = object_asset.data.root_pos_w

    # 转换为环境局部坐标系（减去环境原点偏移）
    object_pos = object_pos_w - env.scene.env_origins

    # 目标位置（环境局部坐标系）
    target_pos = torch.tensor(target_pos_offset, device=env.device).expand(env.num_envs, -1)

    # 计算距离（在环境局部坐标系中）
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


def fall_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    fall_distance: float = 0.12
) -> torch.Tensor:
    """计算掉落惩罚

    Args:
        env: 环境实例
        asset_cfg: 物体资产配置
        fall_distance: 掉落距离阈值

    Returns:
        掉落惩罚 (num_envs,)
    """
    # 获取物体资产
    asset: RigidObject = env.scene[asset_cfg.name]

    # 获取物体位置（世界坐标系）
    object_pos_w = asset.data.root_pos_w

    # 转换为环境局部坐标系（减去环境原点偏移）
    object_pos = object_pos_w - env.scene.env_origins

    # 获取目标位置（环境局部坐标系中的手部附近位置）
    target_pos = torch.tensor([0.0, -0.1, 0.56], device=env.device).expand(env.num_envs, -1)

    # 计算距离
    distance = torch.norm(object_pos - target_pos, p=2, dim=-1)

    # 如果距离超过阈值，返回惩罚
    return torch.where(distance > fall_distance, torch.ones_like(distance), torch.zeros_like(distance))


def pose_diff_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    natural_pose: dict[str, float] | None = None
) -> torch.Tensor:
    """计算手部姿态偏差惩罚 - 鼓励保持接近人手的自然姿态

    Args:
        env: 环境实例
        asset_cfg: 机器人资产配置
        natural_pose: 自然姿态的关节角度字典，如果为None则使用默认值

    Returns:
        姿态偏差惩罚 (num_envs,)
    """
    # 获取机器人资产
    asset: Articulation = env.scene[asset_cfg.name]

    # 定义LeapHand的自然姿态（基于LEAP_Hand_Isaac_Lab项目的官方配置）
    if natural_pose is None:
        # 这些值来自orientation_env.py中的override_default_joint_pos配置
        # 按照ArticulationData的关节索引顺序：a_0到a_15
        natural_joint_angles = [
            0.000,  # a_1
            0.500,  # a_12
            0.000,  # a_5
            0.000,  # a_9
            -0.750, # a_0
            1.300,  # a_13
            0.000,  # a_4
            0.750,  # a_8
            1.750,  # a_2
            1.500,  # a_14
            1.750,  # a_6
            1.750,  # a_10
            0.000,  # a_3
            1.000,  # a_15
            0.000,  # a_7
            0.000,  # a_11
        ]

    # 将自然姿态转换为张量（直接按关节索引顺序）
    natural_joint_pos = torch.tensor(
        natural_joint_angles,
        device=env.device,
        dtype=torch.float32
    ).expand(env.num_envs, -1) # 用于扩展张量的维度，它通过复制数据来创建一个更大的视图，但不会实际分配新的内存

    # 计算当前关节位置与自然姿态的差异
    current_joint_pos = asset.data.joint_pos
    pose_diff = current_joint_pos - natural_joint_pos

    # 计算L2平方惩罚
    pose_diff_penalty = torch.sum(pose_diff ** 2, dim=-1)

    return pose_diff_penalty
