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


def rotation_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    visualize_actual_axis: bool = True,
    target_angular_speed: float = 1.5,
    positive_decay: float = 5.0,
    negative_penalty_weight: float = 2.0,
) -> torch.Tensor:
    """计算旋转速度奖励 - 目标是达到指定的角速度而非越快越好

    Args:
        env: 环境实例
        asset_cfg: 物体资产配置
        visualize_actual_axis: 是否可视化实际旋转轴
        target_angular_speed: 目标角速度大小 (rad/s)
        positive_decay: 正向奖励的指数衰减因子
        negative_penalty_weight: 负向惩罚的权重系数

    Returns:
        旋转速度奖励 (num_envs,)

    Note
    -------
        旋转轴是绕的世界坐标系中的固定轴旋转，而不是绕物体自身的局部坐标系轴旋转
        物体旋转时的旋转轴和Body Frame的表示无关
        
        奖励公式：
        - 正向速度: R = exp(-positive_decay * |projected_velocity - target_angular_speed|)
        - 负向速度: R = negative_penalty_weight * projected_velocity (负惩罚)
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

    # --- 核心修改：使用奖惩一体的逻辑 ---
    # 1. 对于正向速度 (方向正确)
    #    我们使用指数衰减形式，鼓励逼近目标速度
    speed_error_positive = torch.abs(projected_velocity - target_angular_speed)
    # 🔥 限制指数参数，防止exp()溢出
    exp_arg = torch.clamp(-positive_decay * speed_error_positive, min=-10.0, max=10.0)
    reward_positive = torch.exp(exp_arg)

    # 2. 对于负向速度 (方向错误)
    #    我们使用一个线性的惩罚项。速度越负，惩罚越大。
    #    projected_velocity是负的，所以乘以一个正权重就变成了负的奖励（惩罚）
    reward_negative = negative_penalty_weight * projected_velocity

    # 3. 使用 torch.where 根据速度方向选择奖励/惩罚
    #    当 projected_velocity > 0 时，采用 reward_positive
    #    否则 (<= 0)，采用 reward_negative
    reward = torch.where(
        projected_velocity > 0,
        reward_positive,
        reward_negative
    )

    # 🔥 最终的NaN/Inf检查
    # reward = torch.where(torch.isnan(reward) | torch.isinf(reward),
    #                     torch.zeros_like(reward), reward)

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

    # 🔥 指数衰减奖励 - 限制指数参数防止溢出
    exp_arg = torch.clamp(-10.0 * object_dist, min=-10.0, max=10.0)
    reward = torch.exp(exp_arg)

    # 🔥 NaN/Inf检查
    reward = torch.where(torch.isnan(reward) | torch.isinf(reward),
                        torch.zeros_like(reward), reward)

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

class ContinuousRotationSparseReward(ManagerTermBase):
    r"""连续旋转目标达成的稀疏奖励，兼容 ManagerBasedRLEnv 的标准重置流程。

    该奖励函数会在物体绕指定轴旋转超过一定角度时给予奖励，并记录旋转次数。
    奖励值会随着连续成功旋转的次数增加而增加。

    Args:
        env: ManagerBasedRLEnv 环境实例
        asset_cfg (SceneEntityCfg, optional): 物体资产配置，默认为 SceneEntityCfg("object")
        theta_goal (float, optional): 目标旋转角度（弧度），默认为 π/3 (约60度)
        additive_reward (float, optional): 每次成功旋转后累加的奖励值，默认为 0.0
        
    Returns:
        奖励值 (num_envs,)

    Note
    ----
        奖励公式：r = 1.0 + additive_reward * count
            - count: 连续成功旋转的次数，重置时清零
            - additive_reward: 奖励的权重
            - 1.0: 基准奖励值
    """
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._asset: RigidObject = env.scene[self.asset_cfg.name]

        self._theta_goal_default = float(cfg.params.get("theta_goal", 1.0471975512))
        self._additive_reward_default = float(cfg.params.get("additive_reward", 0.0))

        self._norm_eps = 1e-9
        self._angle_eps = 1e-6

        self._reset_internal_buffers()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        q_curr, default_quat = self._read_orientations()
        self._ensure_buffers(q_curr, default_quat)

        env_ids_tensor = self._normalize_env_ids(env_ids)
        reference = self._resolve_reset_reference(default_quat, q_curr)

        self._q_last[env_ids_tensor] = reference[env_ids_tensor]
        self._next_reward[env_ids_tensor] = 1.0
        self._count[env_ids_tensor] = 0
        if self._default_quat_cache is not None:
            self._default_quat_cache[env_ids_tensor] = default_quat[env_ids_tensor]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg | None = None,
        theta_goal: float | None = None,
        additive_reward: float | None = None,
    ) -> torch.Tensor:
        if env is not self._env:
            raise ValueError("ContinuousRotationSparseReward 接收到的 env 与初始化时不一致")

        asset = self._asset if asset_cfg is None else env.scene[asset_cfg.name]
        q_curr = asset.data.root_quat_w.to(device=self.device)
        default_quat = asset.data.default_root_state[:, 3:7].to(device=self.device, dtype=q_curr.dtype)

        self._ensure_buffers(q_curr, default_quat)
        if self._default_quat_cache is not None:
            self._default_quat_cache.copy_(default_quat)

        theta_threshold, additive = self._normalize_parameters(theta_goal, additive_reward)
        target_axis = self._fetch_target_axis(q_curr.dtype)

        theta_signed, theta_abs = self._compute_signed_angle(q_curr, target_axis)
        self._theta_signed = theta_signed
        self._theta_abs = theta_abs

        reward = self._apply_success(theta_signed >= theta_threshold, q_curr, additive)
        return reward

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _reset_internal_buffers(self) -> None:
        self._q_last: torch.Tensor | None = None
        self._next_reward: torch.Tensor | None = None
        self._count: torch.Tensor | None = None
        self._theta_signed: torch.Tensor | None = None
        self._theta_abs: torch.Tensor | None = None
        self._default_quat_cache: torch.Tensor | None = None

    def _read_orientations(self) -> tuple[torch.Tensor, torch.Tensor]:
        q_curr = self._asset.data.root_quat_w.to(device=self.device)
        default_quat = self._asset.data.default_root_state[:, 3:7].to(device=self.device, dtype=q_curr.dtype)
        return q_curr, default_quat

    def _ensure_buffers(self, q_curr: torch.Tensor, default_quat: torch.Tensor) -> None:
        if (
            self._q_last is None
            or self._q_last.shape[0] != self.num_envs
            or self._q_last.device != q_curr.device
            or self._q_last.dtype != q_curr.dtype
        ):
            self._initialize_buffers(q_curr, default_quat)

    def _initialize_buffers(self, q_curr: torch.Tensor, default_quat: torch.Tensor) -> None:
        reference = self._resolve_reset_reference(default_quat, q_curr)

        self._q_last = reference.clone()
        self._next_reward = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self._count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._theta_signed = torch.zeros(self.num_envs, dtype=q_curr.dtype, device=self.device)
        self._theta_abs = torch.zeros(self.num_envs, dtype=q_curr.dtype, device=self.device)
        self._default_quat_cache = default_quat.clone()

    def _normalize_env_ids(self, env_ids: Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _resolve_reset_reference(self, default_quat: torch.Tensor, q_curr: torch.Tensor) -> torch.Tensor:
        default_norm = default_quat / torch.linalg.norm(default_quat, dim=-1, keepdim=True).clamp_min(self._norm_eps)
        current_norm = q_curr / torch.linalg.norm(q_curr, dim=-1, keepdim=True).clamp_min(self._norm_eps)

        alignment = torch.sum(default_norm * current_norm, dim=-1, keepdim=True).abs()
        same_pose = alignment >= (1.0 - 1e-4)
        return torch.where(same_pose, default_norm, current_norm)

    def _normalize_parameters(self, theta_goal: float | None, additive_reward: float | None) -> tuple[float, float]:
        resolved_theta = self._theta_goal_default if theta_goal is None else float(theta_goal)
        resolved_additive = self._additive_reward_default if additive_reward is None else float(additive_reward)
        return resolved_theta, resolved_additive

    def _fetch_target_axis(self, dtype: torch.dtype) -> torch.Tensor:
        if not hasattr(self._env, "command_manager"):
            raise AttributeError("ContinuousRotationSparseReward 需要 command_manager 提供 rotation_axis")

        rotation_axis = self._env.command_manager.get_command("rotation_axis")
        if rotation_axis is None:
            raise ValueError("command_manager.get_command('rotation_axis') 返回 None")

        rotation_axis = rotation_axis.to(device=self.device, dtype=dtype)
        axis_norm = torch.linalg.norm(rotation_axis, dim=-1, keepdim=True).clamp_min(self._angle_eps)
        return rotation_axis / axis_norm

    def _compute_signed_angle(
        self,
        q_curr: torch.Tensor,
        target_axis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quat_diff = quat_mul(q_curr, quat_conjugate(self._q_last))  # type: ignore[arg-type]
        quat_diff = quat_diff / torch.linalg.norm(quat_diff, dim=-1, keepdim=True).clamp_min(self._norm_eps)
        quat_diff = torch.where((quat_diff[:, 0] < 0.0).unsqueeze(-1), -quat_diff, quat_diff)

        vector_part = quat_diff[:, 1:]
        vector_norm = torch.linalg.norm(vector_part, dim=-1, keepdim=True)
        w = torch.clamp(quat_diff[:, 0], -1.0, 1.0)
        theta = 2.0 * torch.atan2(vector_norm.squeeze(-1), w)

        axis = torch.where(vector_norm > self._angle_eps, vector_part / vector_norm, target_axis)
        rot_vec = axis * theta.unsqueeze(-1)
        theta_signed = torch.sum(rot_vec * target_axis, dim=-1)

        return theta_signed, torch.abs(theta_signed)

    def _apply_success(
        self,
        success: torch.Tensor,
        q_curr: torch.Tensor,
        additive_reward: float,
    ) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if torch.any(success):
            reward[success] = self._next_reward[success]
            self._q_last[success] = q_curr[success]
            self._count[success] += 1
            self._next_reward[success] = self._next_reward[success] + additive_reward
        return reward


