# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的终止条件函数"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_falling_termination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    fall_dist: float = 0.12,
    target_pos_offset: tuple[float, float, float] = (0.0, -0.1, 0.56)
) -> torch.Tensor:
    """检查物体是否掉落
    
    Args:
        env: 环境实例
        object_cfg: 物体资产配置
        fall_dist: 掉落距离阈值
        target_pos_offset: 目标位置偏移
        
    Returns:
        物体掉落标志 (num_envs,)
    """
    # 获取物体资产
    object_asset: RigidObject = env.scene[object_cfg.name]
    
    # 计算物体与目标位置的距离
    object_pos = object_asset.data.root_pos_w
    target_pos = torch.tensor(target_pos_offset, device=env.device).expand(env.num_envs, -1)
    object_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    
    return object_dist >= fall_dist
