# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的奖励函数"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_conjugate, quat_mul, wrap_to_pi, quat_from_angle_axis
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_orientation_inv_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_eps: float = 1e-3,
) -> torch.Tensor:
    """方向跟踪奖励 - 使用方向误差的倒数。

    奖励为物体当前姿态与目标姿态之间的方向误差倒数，误差越小奖励越大。

    Args:
        env: ManagerBasedRLEnv - 环境实例
        command_name: str - 命令项名称（用于获取目标姿态）
        object_cfg: SceneEntityCfg - 物体资产配置
        rot_eps: float - 防止除零的小常数（默认 1e-3）

    Returns:
        (num_envs,) 张量，方向跟踪奖励

    NOTE:
        - 奖励公式：R = 1 / (eps + |dtheta|)
    """
    # 获取物体资产
    asset: RigidObject = env.scene[object_cfg.name]

    # 获取目标姿态（从命令管理器）
    # goal_pose 通常是 (pos, quat)，我们取后4维作为目标四元数
    goal_pose = env.command_manager.get_command(command_name)
    goal_quat_w = goal_pose[:, -4:]  # (num_envs, 4) in (w, x, y, z)

    # 计算方向误差（轴角表示的 L2 范数）
    # q_goal ⊖ q_current^(-1) -> 轴角对 -> 角误差（L2范数，单位轴化1，剩下角度）
    dtheta = math_utils.quat_error_magnitude(goal_quat_w, asset.data.root_quat_w)

    # 计算奖励：误差越小，奖励越大
    reward = 1.0 / (dtheta + rot_eps)

    return reward

def success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    threshold: float = 0.05,
) -> torch.Tensor:
    """成功奖励 - 达到目标旋转时的稀疏奖励

    Args:
        env: ManagerBasedRLEnv - 环境实例
        command_name: str - 命令项名称（用于获取目标姿态）
        object_cfg: SceneEntityCfg - 物体资产配置
        threshold: float - 成功容忍度（弧度）

    Returns:
        (num_envs,) 张量，成功奖励
    """
    # 获取物体资产
    asset: RigidObject = env.scene[object_cfg.name]

    # 获取目标姿态（从命令管理器）
    goal_pose = env.command_manager.get_command(command_name)
    goal_quat_w = goal_pose[:, -4:]  # (num_envs, 4) in (w, x, y, z)

    # 计算方向误差（轴角表示的 L2 范数）
    dtheta = math_utils.quat_error_magnitude(goal_quat_w, asset.data.root_quat_w)

    # 计算成功奖励
    success_reward = torch.where(dtheta <= threshold, torch.ones_like(dtheta), torch.zeros_like(dtheta))

    return success_reward

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


###
#  参考LEAP_Hand_Isaac_Lab奖励项
###
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