# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务的课程学习函数
里面的步数指的是全局步数，所有环境累计交互的次数
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import isaaclab.envs.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ============================================================================
# 奖励权重调整课程学习函数
# ============================================================================

def modify_rotation_velocity_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str = "rotation_velocity_reward",
    early_weight: float = 10.0,
    mid_weight: float = 15.0,
    late_weight: float = 20.0,
    mid_step: int = 300_000,
    late_step: int = 800_000
) -> float:
    """
    旋转速度奖励权重调整 - 训练初期低权重，后期逐步提高

    Args:
        env: 环境实例
        env_ids: 环境ID列表
        term_name: 奖励项名称
        early_weight: 初期权重
        mid_weight: 中期权重
        late_weight: 后期权重
        mid_step: 中期开始步数
        late_step: 后期开始步数

    Returns:
        新的奖励权重值
    """
    current_step = env.common_step_counter

    # 确定当前应该使用的权重
    if current_step >= late_step:
        new_weight = late_weight
    elif current_step >= mid_step:
        new_weight = mid_weight
    else:
        new_weight = early_weight

    # 获取当前奖励项配置并更新权重
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if term_cfg.weight != new_weight:
        term_cfg.weight = new_weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

    return new_weight


def modify_rotation_axis_alignment_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str = "rotation_axis_alignment_reward",
    early_weight: float = 1.0,
    mid_weight: float = 0.5,
    late_weight: float = 0.1,
    mid_step: int = 300_000,
    late_step: int = 800_000
) -> float:
    """
    旋转轴对齐奖励权重调整 - 训练初期高权重，后期逐步降低

    Args:
        env: 环境实例
        env_ids: 环境ID列表
        term_name: 奖励项名称
        early_weight: 初期权重
        mid_weight: 中期权重
        late_weight: 后期权重
        mid_step: 中期开始步数
        late_step: 后期开始步数

    Returns:
        新的奖励权重值
    """
    current_step = env.common_step_counter

    # 确定当前应该使用的权重
    if current_step >= late_step:
        new_weight = late_weight
    elif current_step >= mid_step:
        new_weight = mid_weight
    else:
        new_weight = early_weight

    # 获取当前奖励项配置并更新权重
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if term_cfg.weight != new_weight:
        term_cfg.weight = new_weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

    return new_weight


# ============================================================================
# 自适应域随机化课程学习函数
# ============================================================================

def object_mass_adr(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: tuple[float, float],
    enable_step: int = 600_000,
    max_strength_step: int = 1_200_000,
    max_variation: float = 0.5
) -> tuple[float, float]:
    """
    物体质量自适应域随机化 - 修改EventCfg中的mass_distribution_params

    Args:
        env: 环境实例
        env_ids: 环境ID列表
        old_value: 当前的mass_distribution_params值 (min_scale, max_scale)
        enable_step: 启用步数
        max_strength_step: 达到最大强度的步数
        max_variation: 最大变化幅度（相对于1.0的偏差）

    Returns:
        新的mass_distribution_params值 (min_scale, max_scale)
    """
    current_step = env.common_step_counter

    if current_step < enable_step:
        return mdp.modify_env_param.NO_CHANGE

    if current_step >= max_strength_step:
        strength = max_variation
    else:
        progress = (current_step - enable_step) / (max_strength_step - enable_step)
        strength = progress * max_variation

    # 计算新的随机化范围：1.0 ± strength
    min_scale = 1.0 - strength
    max_scale = 1.0 + strength

    return (min_scale, max_scale)


def friction_adr(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: tuple[float, float],
    enable_step: int = 800_000,
    max_strength_step: int = 1_500_000,
    max_variation: float = 0.3
) -> tuple[float, float]:
    """
    摩擦系数自适应域随机化 - 修改EventCfg中的static_friction_range

    Args:
        env: 环境实例
        env_ids: 环境ID列表
        old_value: 当前的static_friction_range值 (min_friction, max_friction)
        enable_step: 启用步数
        max_strength_step: 达到最大强度的步数
        max_variation: 最大变化幅度（相对于1.0的偏差）

    Returns:
        新的static_friction_range值 (min_friction, max_friction)
    """
    current_step = env.common_step_counter

    if current_step < enable_step:
        return mdp.modify_env_param.NO_CHANGE

    if current_step >= max_strength_step:
        strength = max_variation
    else:
        progress = (current_step - enable_step) / (max_strength_step - enable_step)
        strength = progress * max_variation

    # 计算新的随机化范围：1.0 ± strength，确保最小值不小于0.1
    min_friction = max(0.1, 1.0 - strength)
    max_friction = 1.0 + strength

    return (min_friction, max_friction)


def object_scale_adr(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: dict[str, tuple[float, float]],
    enable_step: int = 1_000_000,
    max_strength_step: int = 1_800_000,
    max_variation: float = 0.2
) -> dict[str, tuple[float, float]]:
    """
    物体尺寸自适应域随机化 - 修改EventCfg中的scale_range

    Args:
        env: 环境实例
        env_ids: 环境ID列表
        old_value: 当前的scale_range值 {"x": (min_scale, max_scale), "y": ..., "z": ...}
        enable_step: 启用步数
        max_strength_step: 达到最大强度的步数
        max_variation: 最大变化幅度（相对于1.0的偏差）

    Returns:
        新的scale_range值 {"x": (min_scale, max_scale), "y": ..., "z": ...}
    """
    current_step = env.common_step_counter

    if current_step < enable_step:
        return mdp.modify_env_param.NO_CHANGE

    if current_step >= max_strength_step:
        strength = max_variation
    else:
        progress = (current_step - enable_step) / (max_strength_step - enable_step)
        strength = progress * max_variation

    # 计算新的随机化范围：1.0 ± strength
    min_scale = 1.0 - strength
    max_scale = 1.0 + strength

    # 返回所有轴的随机化范围
    return {
        "x": (min_scale, max_scale),
        "y": (min_scale, max_scale),
        "z": (min_scale, max_scale)
    }


# ============================================================================
# 动作缩放因子调整
# ============================================================================


# ============================================================================
# 旋转轴复杂度课程学习函数
# ============================================================================

def simple_rotation_axis(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: str,
    z_axis_step: int = 0,
    random_axis_step: int = 1_200_000
) -> str:
    """
    简化旋转轴复杂度调整：Z轴 → 任意轴

    Args:
        env: 环境实例
        env_ids: 环境ID列表
        old_value: 当前旋转轴模式
        z_axis_step: Z轴阶段开始步数
        random_axis_step: 任意轴阶段开始步数

    Returns:
        新的旋转轴模式
    """
    current_step = env.common_step_counter

    if current_step >= random_axis_step:
        return "random"
    else:
        return "z_axis"

