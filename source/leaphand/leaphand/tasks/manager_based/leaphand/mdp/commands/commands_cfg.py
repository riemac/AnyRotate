# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .rotation_axis_command import RotationAxisCommand


@configclass
class RotationAxisCommandCfg(CommandTermCfg):
    """旋转轴命令项的配置类。

    更多详细信息请参考 :class:`RotationAxisCommand` 类。
    """

    class_type: type = RotationAxisCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)  # 默认不基于时间重采样

    rotation_axis_mode: str = "z_axis"
    """旋转轴模式。可选项: 'z_axis'(Z轴), 'x_axis'(X轴), 'y_axis'(Y轴), 'random'(随机), 'mixed'(混合)。"""

    change_rotation_axis_interval: int = 0
    """更换旋转轴的间隔步数。0表示不更换。"""

    rotation_axis_noise: float = 0.0
    """添加到旋转轴的噪声。范围: [0.0, 1.0]。"""

    debug_vis: bool = False
    """是否可视化旋转轴命令。"""
