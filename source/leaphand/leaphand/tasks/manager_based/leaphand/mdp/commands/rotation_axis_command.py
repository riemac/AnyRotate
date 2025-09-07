# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""子模块：包含用于旋转轴目标的命令生成器。

该模块定义了 RotationAxisCommand 类，用于为连续旋转任务生成旋转轴命令。
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import RotationAxisCommandCfg


class RotationAxisCommand(CommandTerm):
    """用于连续旋转任务的旋转轴命令项。

    该命令项为基于目标的旋转任务生成3D旋转轴向量。旋转轴遵循右手定则，
    即拇指指向正轴方向，其余手指弯曲方向为正旋转方向。

    命令支持不同的旋转轴模式：
    - "z_axis": 固定Z轴旋转
    - "x_axis": 固定X轴旋转  
    - "y_axis": 固定Y轴旋转
    - "random": 随机轴选择
    - "mixed": 混合模式（包含课程学习）

    与基于时间重采样的典型命令项不同，该命令项可以配置为基于课程进度
    或固定间隔进行重采样。
    """

    cfg: RotationAxisCommandCfg
    """命令项的配置。"""

    def __init__(self, cfg: RotationAxisCommandCfg, env: ManagerBasedRLEnv):
        """初始化命令项类。

        Args:
            cfg: 命令项的配置参数。
            env: 环境对象。
        """
        # 初始化基类
        super().__init__(cfg, env)

        # 创建存储命令的缓冲区
        # -- 旋转轴：(x, y, z) 单位向量
        self.rotation_axis_command = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 轴变化间隔跟踪
        self.axis_change_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # 用于日志记录的指标
        self.metrics["rotation_axis_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["rotation_axis_y"] = torch.zeros(self.num_envs, device=self.device) 
        self.metrics["rotation_axis_z"] = torch.zeros(self.num_envs, device=self.device)

        # 初始化旋转轴
        self._resample_command(torch.arange(self.num_envs, device=self.device))

    def __str__(self) -> str:
        """返回命令项的字符串表示。"""
        msg = "RotationAxisCommand:\n"
        msg += f"\t命令维度: {tuple(self.command.shape[1:])}\n"
        msg += f"\t旋转轴模式: {self.cfg.rotation_axis_mode}\n"
        msg += f"\t重采样时间范围: {self.cfg.resampling_time_range}"
        return msg

    """
    属性
    """

    @property
    def command(self) -> torch.Tensor:
        """期望的旋转轴（单位向量）。形状为 (num_envs, 3)。"""
        return self.rotation_axis_command

    """
    操作方法
    """

    def _update_metrics(self):
        """根据当前状态更新指标。"""
        # 存储当前旋转轴分量用于日志记录
        self.metrics["rotation_axis_x"][:] = self.rotation_axis_command[:, 0]
        self.metrics["rotation_axis_y"][:] = self.rotation_axis_command[:, 1]
        self.metrics["rotation_axis_z"][:] = self.rotation_axis_command[:, 2]

    def _resample_command(self, env_ids: Sequence[int]):
        """为指定环境重采样旋转轴命令。
        
        Args:
            env_ids: 需要重采样命令的环境ID。
        """
        # 如果没有环境需要更新，则直接返回
        if len(env_ids) == 0:
            return
            
        # 获取当前旋转轴模式（考虑课程学习）
        current_mode = self._get_current_axis_mode()
        
        # 根据模式生成旋转轴
        if current_mode == "z_axis":
            # 固定Z轴旋转
            self.rotation_axis_command[env_ids] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        elif current_mode == "x_axis":
            # 固定X轴旋转
            self.rotation_axis_command[env_ids] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        elif current_mode == "y_axis":
            # 固定Y轴旋转
            self.rotation_axis_command[env_ids] = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        elif current_mode == "random":
            # 随机轴选择
            self._sample_random_axis(env_ids)
        elif current_mode == "mixed":
            # 混合模式
            self._sample_mixed_axis(env_ids)
        else:
            # 默认使用Z轴
            self.rotation_axis_command[env_ids] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            
        # 如果指定了噪声，则添加噪声
        if self.cfg.rotation_axis_noise > 0.0:
            self._add_axis_noise(env_ids)
            
        # 重置轴变化计数器
        self.axis_change_counter[env_ids] = 0

    def _update_command(self):
        """根据当前状态更新命令。"""
        # 处理轴变化间隔
        if self.cfg.change_rotation_axis_interval > 0:
            # 增加计数器
            self.axis_change_counter += 1
            # 检查是否达到变化间隔
            change_mask = self.axis_change_counter >= self.cfg.change_rotation_axis_interval
            change_env_ids = change_mask.nonzero(as_tuple=False).squeeze(-1)
            # 如果有环境需要变化轴，则重采样
            if len(change_env_ids) > 0:
                self._resample_command(change_env_ids)

    def _get_current_axis_mode(self) -> str:
        """获取当前旋转轴模式。"""
        return self.cfg.rotation_axis_mode

    def _sample_random_axis(self, env_ids: Sequence[int]):
        """为指定环境采样随机旋转轴。"""
        # 在球面上采样随机单位向量
        random_vecs = torch.randn(len(env_ids), 3, device=self.device)
        random_vecs = random_vecs / torch.norm(random_vecs, dim=-1, keepdim=True)
        self.rotation_axis_command[env_ids] = random_vecs

    def _sample_mixed_axis(self, env_ids: Sequence[int]):
        """采样混合旋转轴（固定轴和随机轴的组合）。"""
        # 25%概率选择每个主要轴，25%概率选择随机轴
        choices = torch.randint(0, 4, (len(env_ids),), device=self.device)
        
        for i, env_id in enumerate(env_ids):
            choice = choices[i].item()
            if choice == 0:
                # X轴
                self.rotation_axis_command[env_id] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            elif choice == 1:
                # Y轴
                self.rotation_axis_command[env_id] = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            elif choice == 2:
                # Z轴
                self.rotation_axis_command[env_id] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
            else:
                # 随机轴
                random_vec = torch.randn(3, device=self.device)
                random_vec = random_vec / torch.norm(random_vec)
                self.rotation_axis_command[env_id] = random_vec

    def _add_axis_noise(self, env_ids: Sequence[int]):
        """向旋转轴添加噪声并重新归一化。"""
        # 生成噪声并添加到旋转轴
        noise = torch.randn(len(env_ids), 3, device=self.device) * self.cfg.rotation_axis_noise
        self.rotation_axis_command[env_ids] += noise
        # 重新归一化为单位向量
        norms = torch.norm(self.rotation_axis_command[env_ids], dim=-1, keepdim=True)
        self.rotation_axis_command[env_ids] = self.rotation_axis_command[env_ids] / norms

    def _set_debug_vis_impl(self, debug_vis: bool):
        """设置旋转轴命令的调试可视化。"""
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")
