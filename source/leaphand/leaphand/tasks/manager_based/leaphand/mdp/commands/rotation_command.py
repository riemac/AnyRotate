# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""连续旋转命令生成器。

该模块实现一个基于固定轴的连续重定向（Continuous Reorientation）命令项。
命令在环境重置时采样初始姿态，并在达到当前目标后沿同一轴持续累积旋转。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import ContinuousRotationCommandCfg


# 预定义的世界坐标系旋转轴映射
_AXIS_MAP = {
    "x": torch.tensor([1.0, 0.0, 0.0]),
    "x_axis": torch.tensor([1.0, 0.0, 0.0]),
    "y": torch.tensor([0.0, 1.0, 0.0]),
    "y_axis": torch.tensor([0.0, 1.0, 0.0]),
    "z": torch.tensor([0.0, 0.0, 1.0]),
    "z_axis": torch.tensor([0.0, 0.0, 1.0]),
}


class ContinuousRotatioCommand(CommandTerm):
    """连续旋转命令项。

    Note
    ----
    命令在世界坐标系下定义：
      - 旋转轴 ``n_w`` 固定在世界坐标系。
      - 每次成功后更新的目标姿态为 ``q_target ← Δq ⊗ q_target``，其中
        ``Δq`` 是绕 ``n_w`` 旋转 ``Δθ`` 的四元数。
    """

    cfg: ContinuousRotationCommandCfg

    def __init__(self, cfg: ContinuousRotationCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object = env.scene[cfg.asset_name]

        # 在环境坐标系保留位置命令，保持与现有观察构建兼容
        init_pos_offset = torch.tensor(cfg.init_pos_offset, dtype=torch.float, device=self.device) # 与手托起物体的设计有关
        self.pos_command_e = self.object.data.default_root_state[:, :3] + init_pos_offset # 环境坐标系  
        self.pos_command_w = self.pos_command_e + self._env.scene.env_origins

        # 目标姿态缓冲（世界坐标系四元数）
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0

        # 每个环境固定的旋转轴、角度增量以及累计统计量
        self.rotation_axis_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.delta_angle = torch.full((self.num_envs,), cfg.delta_angle, device=self.device)
        self.cumulative_rotation = torch.zeros(self.num_envs, device=self.device)
        self.success_counter = torch.zeros(self.num_envs, device=self.device)

        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["cumulative_rotation"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)

        env_ids = torch.arange(self.num_envs, device=self.device)
        self._resample_command(env_ids)

    def __str__(self) -> str:
        msg = "ContinuousRotationCommand:\n"
        msg += f"\t命令维度: {tuple(self.command.shape[1:])}\n"
        msg += f"\t旋转轴: {self.cfg.rotation_axis}\n"
        msg += f"\t角度增量: {self.cfg.delta_angle} rad"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """返回目标位姿 (pos_e, quat_w)。"""

        return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)

    def _update_metrics(self):
        """更新日志指标。"""

        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.quat_command_w
        )
        self.metrics["cumulative_rotation"] = self.cumulative_rotation
        self.metrics["consecutive_success"] = self.success_counter

    def _resample_command(self, env_ids: Sequence[int]):
        """重置命令并采样新的初始目标姿态。"""

        if len(env_ids) == 0:
            return

        axis_key = self.cfg.rotation_axis.lower()
        if axis_key not in _AXIS_MAP:
            raise ValueError(
                f"不支持的旋转轴 '{self.cfg.rotation_axis}'. 支持项为: {sorted(set(_AXIS_MAP.keys()))}."
            )

        axis_vec = _AXIS_MAP[axis_key].to(self.device)
        self.rotation_axis_w[env_ids] = axis_vec

        # 在绕指定轴的随机角度上初始化目标姿态，保证 episode 起点多样性
        random_angle = 2.0 * math.pi * torch.rand(len(env_ids), device=self.device) - math.pi
        axis_batch = axis_vec.repeat(len(env_ids), 1)
        delta_quat = math_utils.quat_from_angle_axis(random_angle, axis_batch)
        base_quat = self.object.data.root_quat_w[env_ids]
        self.quat_command_w[env_ids] = math_utils.quat_mul(delta_quat, base_quat)
        if self.cfg.make_quat_unique:
            self.quat_command_w[env_ids] = math_utils.quat_unique(self.quat_command_w[env_ids])

        self.cumulative_rotation[env_ids] = 0.0
        self.success_counter[env_ids] = 0.0
        self.metrics["orientation_error"][env_ids] = 0.0
        self.metrics["cumulative_rotation"][env_ids] = 0.0
        self.metrics["consecutive_success"][env_ids] = 0.0

    def _update_command(self):
        """根据成功判定沿固定轴增量旋转目标姿态。"""

        if not self.cfg.update_goal_on_success:
            return

        success_mask = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        success_ids = success_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(success_ids) == 0:
            return

        # 成功后沿同一轴推进固定角度，形成连续的目标序列
        delta = torch.full((len(success_ids),), self.cfg.delta_angle, device=self.device)
        delta_quat = math_utils.quat_from_angle_axis(delta, self.rotation_axis_w[success_ids])
        updated = math_utils.quat_mul(delta_quat, self.quat_command_w[success_ids])
        if self.cfg.make_quat_unique:
            updated = math_utils.quat_unique(updated)
        self.quat_command_w[success_ids] = updated

        self.cumulative_rotation[success_ids] += self.cfg.delta_angle
        self.success_counter[success_ids] += 1.0
        self.command_counter[success_ids] += 1
        max_time = self.cfg.resampling_time_range[1]
        self.time_left[success_ids] = max_time

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError("ContinuousRotationCommand 尚未实现调试可视化。")

    def _debug_vis_callback(self, event):
        raise NotImplementedError("ContinuousRotationCommand 尚未实现调试可视化。")
