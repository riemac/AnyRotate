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
    rot = object_asset.data.root_quat_w # 四元数
    
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
