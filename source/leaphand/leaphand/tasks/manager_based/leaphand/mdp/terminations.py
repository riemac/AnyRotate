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


###
# 仿照LEAP_Hand_Sim的实现
###

def object_falling_z_termination(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    z_threshold: float = 0.10,
) -> torch.Tensor:
    """基于 z 轴高度差异判断物体是否掉落（相对于初始高度）

    与 object_fall_penalty 保持相同逻辑：比较当前物体在环境局部坐标系下的 z 与初始 z，
    若高度差超过阈值则判定为掉落（返回 True）。

    Args:
        env: ManagerBasedRLEnv - 环境实例
        object_cfg: SceneEntityCfg - 物体资产配置（默认名为 "object"）
        z_threshold: float - z 轴高度差异阈值，超过此值判定为掉落

    Returns:
        torch.Tensor(bool) - 每个环境是否满足掉落/终止条件 (shape: [num_envs])

    Note:
        - object_pos_w: 物体在世界坐标系下的位置，shape 为 (num_envs, 3)
        - env.scene.env_origins: 每个 env 的世界原点偏置，shape 为 (num_envs, 3)
        - object_pos = object_pos_w - env_origins 得到物体在环境局部坐标系下的位置
        - initial_pos 来自资产 cfg 中的 init_state.pos，表示该物体的初始位置（通常为长度 3 的数组/元组）
        - 比较的是 z 分量：dz = |object_pos[:,2] - initial_pos_z|
    """
    # 获取物体资产对象及其世界坐标位置 (shape: [num_envs, 3])
    asset: RigidObject = env.scene[object_cfg.name]
    object_pos_w = asset.data.root_pos_w  # world positions for each env

    # 将物体位置转换到环境局部坐标系（减去每个 env 的原点偏置）
    # object_pos 的 shape 为 [num_envs, 3]
    env_origins = env.scene.env_origins
    object_pos = object_pos_w - env_origins

    # 从资产配置中读取初始位置（init_state.pos），并转换为与 object_pos 相同的 device/dtype
    # 然后扩展到每个 env（shape: [num_envs, 3]）
    initial_pos = asset.cfg.init_state.pos
    target_pos = torch.tensor(initial_pos, device=env.device, dtype=object_pos.dtype).expand(env.num_envs, -1)

    # 只比较 z 轴高度差（第 2 索引），计算绝对差值并判断是否超过阈值
    dz = torch.abs(object_pos[:, 2] - target_pos[:, 2])

    # 返回布尔张量：对于每个 env，若高度差 >= z_threshold 则视为掉落/终止
    return dz >= z_threshold
    