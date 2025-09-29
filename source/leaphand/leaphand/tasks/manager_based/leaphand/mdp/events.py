# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的事件函数"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_rotation_axis(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    rotation_axis_mode: str = "z_axis",
    rotation_axis_noise: float = 0.05,
) -> None:
    """重置旋转轴
    
    Args:
        env: 环境实例
        env_ids: 需要重置的环境ID
        rotation_axis_mode: 旋转轴模式
        rotation_axis_noise: 旋转轴噪声
    """
    if len(env_ids) == 0:
        return
        
    # 确保环境有rotation_axis属性
    if not hasattr(env, 'rotation_axis'):
        env.rotation_axis = torch.zeros((env.num_envs, 3), dtype=torch.float, device=env.device)
        
    if rotation_axis_mode == "z_axis":
        # 仅绕Z轴旋转
        env.rotation_axis[env_ids] = torch.tensor([0, 0, 1], dtype=torch.float, device=env.device)
    elif rotation_axis_mode == "x_axis":
        # 仅绕X轴旋转
        env.rotation_axis[env_ids] = torch.tensor([1, 0, 0], dtype=torch.float, device=env.device)
    elif rotation_axis_mode == "y_axis":
        # 仅绕Y轴旋转
        env.rotation_axis[env_ids] = torch.tensor([0, 1, 0], dtype=torch.float, device=env.device)
    elif rotation_axis_mode == "random":
        # 随机旋转轴
        random_axes = torch.randn((len(env_ids), 3), dtype=torch.float, device=env.device)
        random_axes = random_axes / torch.norm(random_axes, dim=-1, keepdim=True)
        env.rotation_axis[env_ids] = random_axes
    
    # 添加噪声
    if rotation_axis_noise > 0:
        noise = torch.randn_like(env.rotation_axis[env_ids]) * rotation_axis_noise
        env.rotation_axis[env_ids] += noise
        env.rotation_axis[env_ids] = env.rotation_axis[env_ids] / torch.norm(
            env.rotation_axis[env_ids], dim=-1, keepdim=True
        )


def reset_object_state_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """在指定范围内随机重置物体状态
    
    Args:
        env: 环境实例
        env_ids: 需要重置的环境ID
        pose_range: 位姿范围字典
        velocity_range: 速度范围字典
        asset_cfg: 资产配置
    """
    if len(env_ids) == 0:
        return
        
    # 获取物体资产
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 随机化位置
    if "x" in pose_range:
        asset.data.root_pos_w[env_ids, 0] = sample_uniform(
            pose_range["x"][0], pose_range["x"][1], (len(env_ids),), device=env.device
        )
    if "y" in pose_range:
        asset.data.root_pos_w[env_ids, 1] = sample_uniform(
            pose_range["y"][0], pose_range["y"][1], (len(env_ids),), device=env.device
        )
    if "z" in pose_range:
        asset.data.root_pos_w[env_ids, 2] = sample_uniform(
            pose_range["z"][0], pose_range["z"][1], (len(env_ids),), device=env.device
        )
    
    # 随机化线速度
    if "lin_vel_x" in velocity_range:
        asset.data.root_lin_vel_w[env_ids, 0] = sample_uniform(
            velocity_range["lin_vel_x"][0], velocity_range["lin_vel_x"][1], (len(env_ids),), device=env.device
        )
    if "lin_vel_y" in velocity_range:
        asset.data.root_lin_vel_w[env_ids, 1] = sample_uniform(
            velocity_range["lin_vel_y"][0], velocity_range["lin_vel_y"][1], (len(env_ids),), device=env.device
        )
    if "lin_vel_z" in velocity_range:
        asset.data.root_lin_vel_w[env_ids, 2] = sample_uniform(
            velocity_range["lin_vel_z"][0], velocity_range["lin_vel_z"][1], (len(env_ids),), device=env.device
        )


def randomize_rigid_object_com(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """为刚体对象(RigidObject)随机扰动质心（COM）。

    注意：IsaacLab 内置的 randomize_rigid_body_com 主要面向 Articulation（形状为 (N, num_bodies, 3)）。
    对于 RigidObject，其 get_coms()/set_coms() 形状通常为 (N, 3)。本函数做了相应适配，避免维度索引错误。

    Args:
        env: 训练环境
        env_ids: 需要应用的环境索引；None 表示全部环境
        com_range: 每轴扰动范围，如 {"x":(-0.01,0.01), "y":(-0.01,0.01), "z":(-0.01,0.01)}
        asset_cfg: 目标资产配置，默认是 "object"
    """
    # 解析环境索引到 CPU（PhysX 接口期望 CPU 张量）
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # 获取对象
    obj: RigidObject = env.scene[asset_cfg.name]

    # 读取并克隆当前 COM，形状期望为 (num_envs, 3)
    coms = obj.root_physx_view.get_coms().clone()
    if coms.ndim != 2 or coms.shape[-1] != 3:
        # 若形状异常，直接返回（保守回退）
        return

    # 采样随机扰动（逐轴范围）
    lows = torch.tensor(
        [com_range.get("x", (0.0, 0.0))[0], com_range.get("y", (0.0, 0.0))[0], com_range.get("z", (0.0, 0.0))[0]],
        device="cpu",
    )
    highs = torch.tensor(
        [com_range.get("x", (0.0, 0.0))[1], com_range.get("y", (0.0, 0.0))[1], com_range.get("z", (0.0, 0.0))[1]],
        device="cpu",
    )
    noise = lows + torch.rand((len(env_ids), 3), device="cpu") * (highs - lows)

    # 应用扰动到选定环境
    coms[env_ids] += noise
    obj.root_physx_view.set_coms(coms, env_ids)

