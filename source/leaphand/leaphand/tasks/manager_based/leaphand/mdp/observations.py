# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的观测函数"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rotation_axis(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """返回当前任务的目标旋转轴
    
    Args:
        env: 环境实例
        asset_cfg: 资产配置（未使用，保持接口一致性）
        
    Returns:
        当前任务的目标旋转轴 (num_envs, 3)
    """
    # 如果环境还没有rotation_axis属性，则初始化
    if not hasattr(env, 'rotation_axis'):
        env.rotation_axis = torch.zeros((env.num_envs, 3), dtype=torch.float, device=env.device)
        # 默认设置为Z轴旋转
        env.rotation_axis[:, 2] = 1.0
    
    return env.rotation_axis


def relative_fingertip_positions(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fingertip_body_names: list[str] | None = None,
) -> torch.Tensor:
    """获取指尖位置（相对于环境原点）

    基于原始LEAP_Hand_Isaac_Lab项目的实现方式

    Args:
        env: 环境实例
        robot_cfg: 机器人资产配置
        fingertip_body_names: 指尖体名称列表

    Returns:
        指尖位置 (num_envs, num_fingertips * 3)
    """
    # 获取机器人资产
    robot: Articulation = env.scene[robot_cfg.name]

    # 默认指尖体名称（与原始项目保持一致）
    if fingertip_body_names is None:
        fingertip_body_names = ["fingertip", "thumb_fingertip", "fingertip_2", "fingertip_3"]

    # 获取指尖body索引
    try:
        fingertip_indices = []
        for name in fingertip_body_names:
            if name in robot.body_names:
                fingertip_indices.append(robot.body_names.index(name))
            else:
                # 如果找不到，尝试查找类似的名称
                similar_names = [body_name for body_name in robot.body_names if name.lower() in body_name.lower()]
                if similar_names:
                    print(f"警告: 找不到指尖body '{name}'，使用相似名称 '{similar_names[0]}'")
                    fingertip_indices.append(robot.body_names.index(similar_names[0]))
                else:
                    print(f"警告: 找不到指尖body '{name}'，跳过")

        if not fingertip_indices:
            print("错误: 没有找到任何有效的指尖body，返回零向量")
            print(f"可用的body名称: {robot.body_names}")
            return torch.zeros((env.num_envs, 12), device=env.device)  # 4个指尖 * 3维度

        # 获取指尖位置（世界坐标）
        fingertip_pos = robot.data.body_pos_w[:, fingertip_indices]  # (num_envs, num_fingertips, 3)

        # 转换为相对于环境原点的位置（与原始项目保持一致）
        num_fingertips = len(fingertip_indices)
        env_origins_expanded = env.scene.env_origins.repeat((1, num_fingertips)).reshape(
            env.num_envs, num_fingertips, 3
        )
        fingertip_pos_relative = fingertip_pos - env_origins_expanded

        # 展平为 (num_envs, num_fingertips * 3)
        return fingertip_pos_relative.view(env.num_envs, -1)

    except Exception as e:
        print(f"计算指尖位置时出错: {e}")
        print(f"可用的body名称: {robot.body_names}")
        return torch.zeros((env.num_envs, 12), device=env.device)


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
    rot = object_asset.data.root_quat_w
    
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
