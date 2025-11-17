# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import math
from pickle import NONE

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkersCfg

from .rotation_command import ContinuousRotationCommand


@configclass
class ContinuousRotationCommandCfg(CommandTermCfg):
    """连续旋转命令配置。"""

    class_type: type = ContinuousRotationCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)

    asset_name: str = MISSING
    """参与重定向的物体在场景中的名称。"""

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """相对于物体默认根姿态的位置偏移。"""

    rotation_axis: str = "z"
    """连续旋转所围绕的世界坐标系轴（x/y/z）。"""

    delta_angle: float = math.pi / 8.0
    """每次成功后的增量旋转角度（单位: rad）。"""

    make_quat_unique: bool = True
    """是否将目标四元数约束为唯一表示。"""

    orientation_success_threshold: float = NONE  # 改为MISSING会报错，因为它要求必须提供值
    """判定完成当前目标姿态的角度阈值（单位: rad）。"""

    update_goal_on_success: bool = True
    """是否在成功达到目标后沿轴继续更新目标。"""
    
    def __post_init__(self):
        """初始化后处理，根据 delta_angle 自动计算成功阈值（5%容差）"""
        # 如果未提供成功阈值，则根据 delta_angle 自动计算
        if self.orientation_success_threshold == NONE:
            # 参考 DirectRLEnv 实现，允许约 0.2rad 的姿态误差，同时兼容更大旋转步长
            self.orientation_success_threshold = max(0.2, self.delta_angle / 2.0)
        # print(f"成功阈值: {self.orientation_success_threshold}")