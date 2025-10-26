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
    """检查物体是否掉落（物体位置先相对于env_origins再与目标位置比较）

    计算步骤与原逻辑等价，但先将物体位置变换为相对于环境原点再计算距离。

    Args:
        env: ManagerBasedRLEnv - 环境实例
        object_cfg: SceneEntityCfg - 物体资产配置（默认名为 "object"）
        fall_dist: float - 距离阈值，超过则判定为掉落
        target_pos_offset: tuple[float, float, float] - 目标位置相对于环境原点的偏移
    Returns:
        torch.Tensor (bool) - 每个环境是否满足掉落/终止条件 (shape: [num_envs])

    Note:
        令 object_pos_world 为物体在世界坐标系下的位置，env_origins 为每个环境的原点偏置，
        则相对于环境原点的物体位置：
            object_pos_rel = object_pos_world - env_origins

        目标位置已以环境原点为参考：
            target_pos = target_pos_offset

        欧氏距离：
            dist = || object_pos_rel - target_pos ||_2

        掉落判定（终止）：
            terminated = dist >= fall_dist
    """
    # 获取物体资产与世界坐标位置 (shape: [num_envs, 3])
    object_asset: RigidObject = env.scene[object_cfg.name]
    object_pos_world = object_asset.data.root_pos_w

    # 获取每个环境的原点 (shape: [num_envs, 3])
    env_origins = env.scene.env_origins

    # 将物体位置变为相对于环境原点的坐标
    object_pos = object_pos_world - env_origins

    # 目标位置直接使用相对于环境原点的偏移，扩展到每个env
    target_pos_offset_tensor = torch.tensor(
        target_pos_offset, device=env.device, dtype=object_pos.dtype
    )
    target_pos = target_pos_offset_tensor.unsqueeze(0).expand(env.num_envs, -1)

    # 计算欧氏距离并判断是否掉落
    object_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    return object_dist >= fall_dist

def object_away_from_robot(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if object has gone far from the robot.

    The object is considered to be out-of-reach if the distance between the robot and the object is greater
    than the threshold.

    Args:
        env: The environment object.
        threshold: The threshold for the distance between the robot and the object.
        asset_cfg: The configuration for the robot entity. Default is "robot".
        object_cfg: The configuration for the object entity. Default is "object".
    """
    # extract useful elements
    robot = env.scene[asset_cfg.name]
    object = env.scene[object_cfg.name]

    # compute distance
    dist = torch.norm(robot.data.root_pos_w - object.data.root_pos_w, dim=1)

    return dist > threshold
    