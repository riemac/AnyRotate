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
import weakref

from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_from_angle_axis

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

        # 可视化相关
        self._is_visualizing = False
        self._debug_vis_handle = None

        # 如果配置了debug_vis，则启用可视化
        if self.cfg.debug_vis:
            self.set_debug_vis(True)

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

    def _resample_command(self, env_ids: Sequence[int]): # 生成新的任务目标，保持训练多样性
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
        # 检查是否启用可视化
        if not self.cfg.visualizer_cfg.enabled:
            return

        # 设置可视化标记器的可见性
        if debug_vis:
            if not hasattr(self, "rotation_axis_visualizer"):
                # 使用配置文件中定义的可视化配置
                self.rotation_axis_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg.marker_cfg)
            # 设置可见性为真
            self.rotation_axis_visualizer.set_visibility(True)
        else:
            if hasattr(self, "rotation_axis_visualizer"):
                self.rotation_axis_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """旋转轴可视化回调函数。

        为每个环境显示旋转轴箭头，箭头位置跟随物体并正确处理env_origins偏移，
        箭头方向指向各自环境的旋转轴方向。
        """
        # 检查是否有可视化器
        if not hasattr(self, "rotation_axis_visualizer"):
            return

        # 获取物体位置（假设物体在场景中的名称为"object"）
        try:
            # 从环境中获取物体位置
            object_asset = self._env.scene["object"]
            object_pos_w = object_asset.data.root_pos_w

            # 计算箭头位置（物体上方，便于观察）
            # object_pos_w已经包含了env_origins偏移，所以直接使用即可
            arrow_positions = object_pos_w.clone()
            arrow_positions[:, 2] += self.cfg.visualizer_cfg.offset_above_object

            # 计算箭头方向（基于每个环境的旋转轴方向）
            arrow_orientations = self._compute_arrow_orientations()

            # 创建marker_indices - 参考官方示例
            # 所有箭头都使用同一个原型（索引0）
            all_envs = torch.arange(self.num_envs, device=self.device)
            marker_indices = torch.zeros_like(all_envs)

            # 更新可视化 - 为每个环境显示一个箭头
            # 参考官方示例的调用方式
            self.rotation_axis_visualizer.visualize(
                translations=arrow_positions,
                orientations=arrow_orientations,
                marker_indices=marker_indices
            )
        except Exception:
            # 如果获取物体位置失败，则不显示可视化
            pass

    def _compute_arrow_orientations(self) -> torch.Tensor:
        """计算箭头的方向四元数。

        箭头默认指向X轴正方向，需要旋转到旋转轴方向。
        遵循右手螺旋定则：拇指指向旋转轴正方向，其余手指弯曲方向为正旋转方向。

        Returns:
            形状为 (num_envs, 4) 的四元数张量 (w, x, y, z)
        """
        # 箭头默认方向（X轴正方向）
        default_direction = torch.tensor([1.0, 0.0, 0.0], device=self.device)

        # 目标方向（旋转轴方向）
        target_directions = self.rotation_axis_command

        # 使用更简化的方法计算箭头方向
        orientations = self._compute_arrow_orientations_optimized(target_directions)

        return orientations

    def _compute_arrow_orientations_optimized(self, target_directions: torch.Tensor) -> torch.Tensor:
        """优化的箭头方向计算方法

        Args:
            target_directions: 目标方向向量 (num_envs, 3)

        Returns:
            旋转四元数 (num_envs, 4) - (w, x, y, z)
        """
        # 箭头默认方向（X轴正方向）
        default_direction = torch.tensor([1.0, 0.0, 0.0], device=self.device)

        # 归一化目标方向
        target_norm = torch.norm(target_directions, dim=-1, keepdim=True)
        target_normalized = torch.where(target_norm > 1e-6, target_directions / target_norm, default_direction.unsqueeze(0))

        # 计算旋转轴（叉积）和角度（点积）
        rotation_axis = torch.cross(default_direction.unsqueeze(0).expand_as(target_normalized), target_normalized, dim=-1)
        cos_angle = torch.sum(default_direction.unsqueeze(0) * target_normalized, dim=-1)

        # 计算旋转角度
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))

        # 处理旋转轴为零的情况（向量平行）
        rotation_axis_norm = torch.norm(rotation_axis, dim=-1, keepdim=True)
        rotation_axis = torch.where(
            rotation_axis_norm > 1e-6,
            rotation_axis / rotation_axis_norm,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).expand_as(rotation_axis)
        )

        # 使用Isaac Lab官方函数计算四元数
        orientations = quat_from_angle_axis(angle, rotation_axis)

        return orientations

    @property
    def has_debug_vis_implementation(self) -> bool:
        """检查是否实现了调试可视化。"""
        return True

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """设置调试可视化状态。

        Args:
            debug_vis: 是否启用调试可视化。

        Returns:
            是否成功设置调试可视化。
        """
        # 检查是否支持调试可视化
        if not self.has_debug_vis_implementation:
            return False
        # 切换调试可视化对象
        self._set_debug_vis_impl(debug_vis)
        # 切换调试可视化标志
        self._is_visualizing = debug_vis
        # 切换调试可视化句柄
        if debug_vis:
            # 如果不存在，则为post update事件创建订阅者
            if self._debug_vis_handle is None:
                import omni.kit.app
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # 移除调试可视化句柄
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        return True
