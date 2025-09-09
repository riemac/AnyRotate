# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg

from .rotation_axis_command import RotationAxisCommand


@configclass
class RotationAxisVisualizerCfg:
    """旋转轴可视化配置类

    参考Isaac Lab官方Commands的可视化配置。
    """

    # 基础参数 TODO:是否和debug_vis重复?
    enabled: bool = True
    """是否启用旋转轴可视化"""

    # 位置参数
    offset_above_object: float = 0.15
    """目标旋转轴箭头在物体上方的偏移距离（米）"""

    # 可视化标记器配置
    marker_cfg: VisualizationMarkersCfg = None
    """箭头可视化标记器配置，将在__post_init__中设置"""

    def __post_init__(self):
        """初始化后处理，设置可视化标记器配置"""
        from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG

        # 使用Isaac Lab提供的标准红色箭头配置（目标旋转轴）
        self.marker_cfg = RED_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/target_rotation_axis"
        )
        # 设置箭头尺寸
        self.marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.3)


@configclass
class RotationAxisCommandCfg(CommandTermCfg):
    """旋转轴命令项的配置类。

    更多详细信息请参考 :class:`RotationAxisCommand` 类。
    """

    class_type: type = RotationAxisCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)  # 默认不基于时间重采样

    rotation_axis_mode: str = "z_axis" # 课程学习会动态调整，默认Z轴
    """旋转轴模式。可选项: 'z_axis'(Z轴), 'x_axis'(X轴), 'y_axis'(Y轴), 'random'(随机)。"""

    change_rotation_axis_interval: int = 0
    """更换旋转轴的间隔步数。0表示不更换。"""

    rotation_axis_noise: float = 0.0
    """添加到旋转轴的噪声。范围: [0.0, 1.0]。"""

    debug_vis: bool = False
    """是否可视化旋转轴命令。"""

    visualizer_cfg: RotationAxisVisualizerCfg = RotationAxisVisualizerCfg()
    """旋转轴可视化配置。"""
