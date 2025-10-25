# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的奖励函数"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_conjugate, quat_mul, wrap_to_pi, quat_from_angle_axis
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fingertip_distance_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """计算指尖到物体中心距离的惩罚 - 鼓励指尖接近物体

    Args:
        env: 环境实例
        object_cfg: 物体资产配置
        robot_cfg: 机器人资产配置

    Returns:
        指尖距离惩罚 (num_envs,)
        
    Note:
        奖励公式：R = mean(||fingertip_pos - object_pos||_2)
        其中fingertip_pos是每个指尖的世界坐标位置，object_pos是物体质心的世界坐标位置
        使用平均距离而不是最小距离可以让所有手指都参与抓取，避免部分手指不操作
    """
    # 获取物体资产
    object_asset: RigidObject = env.scene[object_cfg.name]
    robot_asset: Articulation = env.scene[robot_cfg.name]

    # 获取物体质量中心位置
    object_pos_w = object_asset.data.root_pos_w

    # LeapHand指尖body名称（基于实际的body_names输出）
    fingertip_body_names = [
        "fingertip",         # 食指指尖
        "thumb_fingertip",   # 拇指指尖
        "fingertip_2",       # 中指指尖
        "fingertip_3"        # 无名指指尖
    ]

    # 获取所有指尖的位置
    fingertip_distances = []
    
    for body_name in fingertip_body_names:
        # 获取指尖body的世界坐标位置
        body_indices, _ = robot_asset.find_bodies(body_name)
        # 若无匹配，直接抛出异常
        if len(body_indices) == 0:
            raise IndexError(f"Body not found: {body_name}")
        # 使用第一个匹配到的索引（Python int）
        body_idx = int(body_indices[0])        
        fingertip_pos_w = robot_asset.data.body_pos_w[:, body_idx]
        
        # 计算指尖到物体中心的距离
        distance = torch.norm(fingertip_pos_w - object_pos_w, p=2, dim=-1)
        fingertip_distances.append(distance)

    # 将所有指尖距离堆叠为张量 (num_envs, num_fingertips)
    fingertip_distances_tensor = torch.stack(fingertip_distances, dim=-1)
    # 计算最小距离（最接近物体的指尖）或平均距离（更平滑）
    # min_distance = torch.min(fingertip_distances_tensor, dim=-1)[0]
    distance = torch.mean(fingertip_distances_tensor, dim=-1)
    # 返回形状 (num_envs,)
    return distance


def rotation_axis_alignment_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    theta_tolerance: float = 0.1,
    decay_factor: float = 5.0,
) -> torch.Tensor:
    """计算旋转轴容忍差奖励 - 指数衰减型奖励

    Args:
        env: 环境实例
        asset_cfg: 物体资产配置
        theta_tolerance: 角度容忍度 (弧度)
        decay_factor: 指数衰减因子

    Returns:
        旋转轴对齐奖励 (num_envs,)

    Note:
        奖励公式：R_axis = weight * exp(-decay_factor * max(0, theta - theta_tolerance))
        其中theta是实际旋转轴与目标旋转轴之间的夹角
    """
    # 获取物体资产
    asset: RigidObject = env.scene[asset_cfg.name]

    # 获取当前物体旋转
    current_object_rot = asset.data.root_quat_w

    # 初始化独立的last_object_rot状态（避免与rotation_velocity_reward冲突）
    if not hasattr(env, 'last_object_rot_alignment'):
        env.last_object_rot_alignment = torch.zeros((env.num_envs, 4), dtype=torch.float, device=env.device)
        env.last_object_rot_alignment[:, 0] = 1.0  # 初始化为单位四元数

    # 计算旋转差异
    quat_diff = quat_mul(current_object_rot, quat_conjugate(env.last_object_rot_alignment))

    # 计算实际旋转轴
    axis = quat_diff[:, 1:4]
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    valid_rotation = axis_norm.squeeze(-1) > 1e-6
    axis = torch.where(valid_rotation.unsqueeze(-1), axis / axis_norm, torch.zeros_like(axis))

    # 获取目标旋转轴
    target_axis = env.command_manager.get_command("rotation_axis")

    # 计算实际旋转轴与目标旋转轴之间的夹角
    # 使用点积计算夹角：cos(theta) = a·b / (|a||b|)
    dot_product = torch.sum(axis * target_axis, dim=-1)
    # 🔥 更严格的数值稳定性处理
    dot_product = torch.clamp(dot_product, -0.9999, 0.9999)  # 避免acos(±1)的数值问题

    # 🔥 安全的角度计算，处理无效旋转的情况
    theta = torch.where(
        valid_rotation,
        torch.acos(torch.abs(dot_product)),  # 只计算绝对值的角度，避免方向问题
        torch.zeros_like(dot_product)        # 无效旋转时角度为0
    )

    # 计算指数衰减奖励
    angle_error = torch.clamp(theta - theta_tolerance, min=0.0)
    reward = torch.exp(-decay_factor * angle_error)

    # 🔥 额外的NaN检查和处理
    reward = torch.where(torch.isnan(reward) | torch.isinf(reward),
                        torch.zeros_like(reward), reward)

    # 更新上一帧旋转（独立状态）
    env.last_object_rot_alignment[:] = current_object_rot.clone()

    return reward


def unstable_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """计算稳定性惩罚 - 减少不必要的震荡

    Args:
        env: 环境实例
        object_cfg: 物体资产配置

    Returns:
        稳定性惩罚 (num_envs,)，实际为负的惩罚项
        
    Note:
        奖励公式：R = weight * ||v||_2
        其中v是物体的线速度向量，使用L2范数计算速度大小
        weight是负号，表示这是一个惩罚项，速度越大惩罚越大
    """
    # 获取物体资产
    object_asset: RigidObject = env.scene[object_cfg.name]

    # 基于物体线速度的稳定性惩罚 TODO：使用质心速度
    object_lin_vel = object_asset.data.root_lin_vel_w
    # object_lin_vel = object_asset.data.body_com_lin_vel_w.squeeze(-2)
    penalty = torch.norm(object_lin_vel, p=2, dim=-1)

    return penalty


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

###
#  参考LEAP_Hand_Sim奖励项
###

def rotate_angvel_clipped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    clip_min: float = -0.25,
    clip_max: float = 0.25,
) -> torch.Tensor:
    """计算物体角速度在目标轴上的投影并裁剪。

    Args:
        env: ManagerBasedRLEnv
            环境实例
        asset_cfg: SceneEntityCfg
            物体资产配置，默认为SceneEntityCfg("object")
        clip_min: float
            裁剪下限，默认为-0.25
        clip_max: float
            裁剪上限，默认为0.25

    Returns:
        裁剪后的角速度投影奖励 (num_envs,)

    NOTE:
        奖励公式：r = clip(dot(ω_w, â), clip_min, clip_max)
        其中:
        - ω_w: 物体在世界坐标系下的角速度向量
        - â: 目标旋转轴的单位向量
        - clip(): 裁剪函数，将值限制在[clip_min, clip_max]范围内
    """
    object_asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_w = object_asset.data.root_ang_vel_w  # (N, 3)
    target_axis = env.command_manager.get_command("rotation_axis")  # (N, 3)
    # 投影到目标轴（假设target_axis已归一化，否则需归一化）
    proj = torch.sum(ang_vel_w * target_axis, dim=-1)  # (N,)
    # 可选：只奖励同向旋转
    proj = torch.where(proj > 0, proj, torch.zeros_like(proj))
    return torch.clamp(proj, min=clip_min, max=clip_max)


def object_linvel_l1_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """计算物体线速度的L1范数惩罚。

    Args:
        env: 环境实例
        object_cfg: 物体资产配置

    Returns:
        线速度惩罚 (num_envs,)

    Note:
        惩罚公式：P = ∑|v_w|
        其中v_w是物体在世界坐标系下的线速度向量
        使用L1范数可以分别惩罚各个方向的速度分量
    """
    object_asset: RigidObject = env.scene[object_cfg.name]
    lin_vel_w = object_asset.data.root_lin_vel_w  # (N, 3)
    return torch.sum(torch.abs(lin_vel_w), dim=-1)


def work_penalty_squared(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机械功惩罚（平方项）

    Args:
        env: 环境实例
        robot_cfg: 机器人资产配置

    Returns:
        机械功惩罚 (num_envs,)

    Note:
        数学公式：w = (∑_j τ_j · q̇_j)^2 
        其中τ_j为第j个关节的力矩，q̇_j为第j个关节的速度。
        该项鼓励在实现目标的同时降低做功（抑制无效挤压/抖动）。
    """
    robot: Articulation = env.scene[robot_cfg.name]
    tau = robot.data.applied_torque[:, robot_cfg.joint_ids]  # (num_envs, DoF)
    qd = robot.data.joint_vel[:, robot_cfg.joint_ids]        # (num_envs, DoF)
    power = torch.sum(tau * qd, dim=-1)  # (num_envs,)
    return torch.square(power)


def object_fall_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    z_threshold: float = 0.10
) -> torch.Tensor:
    """计算物体掉落惩罚 - 基于z轴高度差异判断

    Args:
        env: 环境实例
        asset_cfg: 物体资产配置
        z_threshold: z轴高度差异阈值，超过此值判定为掉落

    Returns:
        掉落惩罚 (num_envs,)，掉落时为1，否则为0

    Note:
        惩罚公式：P = [|z - z_init| > threshold]
        其中z是当前物体高度，z_init是初始高度
        使用初始位置作为参考，避免手部移动导致的误判
    """
    # 获取物体资产
    asset: RigidObject = env.scene[asset_cfg.name]

    # 获取物体位置（世界坐标系）
    object_pos_w = asset.data.root_pos_w

    # 转换为环境局部坐标系（减去环境原点偏移）
    object_pos = object_pos_w - env.scene.env_origins

    # 获取目标位置（环境局部坐标系中的手部附近位置）
    inital_pos = asset.cfg.init_state.pos
    target_pos = torch.tensor(inital_pos, device=env.device).expand(env.num_envs, -1)

    # 计算距离
    dz = torch.abs(object_pos[:, 2] - target_pos[:, 2])

    # 如果距离超过阈值，返回惩罚
    return torch.where(dz > z_threshold, torch.ones_like(dz), torch.zeros_like(dz))


###
#  参考LEAP_Hand_Isaac_Lab奖励项
###

