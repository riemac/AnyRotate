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
from isaaclab.utils.math import quat_conjugate, quat_mul, wrap_to_pi, quat_from_angle_axis
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rotation_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    visualize_actual_axis: bool = True,
    target_angular_speed: float = 1.5,
    decay_factor: float = 5.0,
) -> torch.Tensor:
    """计算旋转速度奖励 - 目标是达到指定的角速度而非越快越好

    Args:
        env: 环境实例
        asset_cfg: 物体资产配置
        visualize_actual_axis: 是否可视化实际旋转轴
        target_angular_speed: 目标角速度大小 (rad/s)
        decay_factor: 指数衰减因子

    Returns:
        旋转速度奖励 (num_envs,)

    Note:
        旋转轴是绕的世界坐标系中的固定轴旋转，而不是绕物体自身的局部坐标系轴旋转
        物体旋转时的旋转轴和Body Frame的表示无关

        奖励公式：R = exp(-decay_factor * |projected_velocity - target_angular_speed|)
    
    TODO:
        计划修改为奖惩一体型的，当speed_error为负，给予惩罚。speed_error为正，给予奖励
    """
    # 获取物体资产
    asset: RigidObject = env.scene[asset_cfg.name]

    # 获取当前物体旋转
    current_object_rot = asset.data.root_quat_w # 固定的世界坐标系

    # 初始化last_object_rot如果不存在
    if not hasattr(env, 'last_object_rot'):
        env.last_object_rot = torch.zeros((env.num_envs, 4), dtype=torch.float, device=env.device)
        env.last_object_rot[:, 0] = 1.0  # 初始化为单位四元数

    # 计算旋转差异
    quat_diff = quat_mul(current_object_rot, quat_conjugate(env.last_object_rot))
    angle = 2.0 * torch.acos(torch.clamp(torch.abs(quat_diff[:, 0]), max=1.0))

    # 计算旋转轴
    axis = quat_diff[:, 1:4]
    axis_norm = torch.norm(axis, dim=-1, keepdim=True) # 计算旋转轴的范数
    valid_rotation = axis_norm.squeeze(-1) > 1e-6 #  # 判断是否为有效旋转(范数大于阈值) omega^hat*theta，若物体静止不动，该范式将非常小
    axis = torch.where(valid_rotation.unsqueeze(-1), axis / axis_norm, torch.zeros_like(axis)) # 对有效旋转进行归一化,无效旋转置零

    # 获取目标旋转轴 - 从Command管理器获取
    rotation_axis = env.command_manager.get_command("rotation_axis")

    # 计算沿指定旋转轴的角速度
    angular_velocity = angle / env.step_dt
    projected_velocity = torch.sum(axis * rotation_axis, dim=-1) * angular_velocity

    # 更新上一帧旋转
    env.last_object_rot[:] = current_object_rot.clone()

    # 计算目标角速度奖励 - 使用指数衰减型奖励
    speed_error = torch.abs(projected_velocity - target_angular_speed) # 目标是让projected_velocity（带方向）接近target_angular_speed
    reward = torch.exp(-decay_factor * speed_error)

    # 可视化实际旋转轴
    if visualize_actual_axis:
        _visualize_actual_rotation_axis(env, asset, axis, valid_rotation)

    return reward


def _visualize_actual_rotation_axis(
    env: ManagerBasedRLEnv,
    asset: RigidObject,
    actual_axis: torch.Tensor,
    valid_rotation: torch.Tensor,
):
    """可视化实际旋转轴

    Args:
        env: 环境实例
        asset: 物体资产
        actual_axis: 实际旋转轴 (num_envs, 3)
        valid_rotation: 有效旋转掩码 (num_envs,)
    """
    # 初始化可视化器（如果不存在）
    if not hasattr(env, '_actual_axis_visualizer'):
        # 创建蓝色箭头可视化器
        marker_cfg = BLUE_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Reward/actual_rotation_axis"
        )
        # 设置箭头尺寸（与目标轴相同）
        marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.3)
        env._actual_axis_visualizer = VisualizationMarkers(marker_cfg)

    # 只显示有效旋转的箭头
    valid_env_ids = valid_rotation.nonzero(as_tuple=False).squeeze(-1)
    if len(valid_env_ids) == 0:
        return

    # 获取物体位置
    object_pos_w = asset.data.root_pos_w[valid_env_ids]

    # 计算箭头位置（物体上方，但与目标轴有不同偏移避免重叠）
    arrow_positions = object_pos_w.clone()
    arrow_positions[:, 2] += 0.20  # 比目标轴稍高一些（目标轴是0.15）

    # 直接使用实际旋转轴计算箭头方向
    valid_axes = actual_axis[valid_env_ids]
    arrow_orientations = _compute_arrow_orientations_from_axis(valid_axes, env.device)

    # 创建marker_indices
    marker_indices = torch.zeros(len(valid_env_ids), device=env.device, dtype=torch.int32)

    # 更新可视化
    env._actual_axis_visualizer.visualize(
        translations=arrow_positions,
        orientations=arrow_orientations,
        marker_indices=marker_indices
    )


def _compute_arrow_orientations_from_axis(axis: torch.Tensor, device: torch.device) -> torch.Tensor:
    """从旋转轴计算箭头方向四元数

    Args:
        axis: 已归一化的旋转轴向量 (num_envs, 3)
        device: 设备

    Returns:
        箭头方向四元数 (num_envs, 4) - (w, x, y, z)

    Note:
        输入的axis已经在rotation_velocity_reward中被归一化，无需重复归一化
    """
    # 箭头默认方向（X轴正方向）
    default_direction = torch.tensor([1.0, 0.0, 0.0], device=device)

    # 计算旋转轴（叉积）和角度（点积）
    rotation_axis = torch.cross(default_direction.unsqueeze(0).expand_as(axis), axis, dim=-1)
    cos_angle = torch.sum(default_direction.unsqueeze(0) * axis, dim=-1)

    # 计算旋转角度
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

    # 处理旋转轴为零的情况（向量平行）
    rotation_axis_norm = torch.norm(rotation_axis, dim=-1, keepdim=True)
    rotation_axis = torch.where(
        rotation_axis_norm > 1e-6,
        rotation_axis / rotation_axis_norm,
        torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).expand_as(rotation_axis)
    )

    # 使用Isaac Lab官方函数计算四元数
    orientations = quat_from_angle_axis(angle, rotation_axis)

    return orientations


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
    # 限制点积值在[-1, 1]范围内，避免数值误差
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    # 计算夹角（取绝对值，因为我们关心的是对齐程度）
    theta = torch.acos(torch.abs(dot_product)) # TODO: 这里用绝对值可能有问题，会导致轴重合但方向相反的情况得到奖励

    # 计算指数衰减奖励
    angle_error = torch.clamp(theta - theta_tolerance, min=0.0)
    reward = torch.exp(-decay_factor * angle_error)

    # 更新上一帧旋转（独立状态）
    env.last_object_rot_alignment[:] = current_object_rot.clone()

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

    # 基于物体线速度的稳定性惩罚
    object_lin_vel = object_asset.data.root_lin_vel_w
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
