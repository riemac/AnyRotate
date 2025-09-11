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

def modify_action_scale_factor(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    action_term_name: str = "hand_joint_pos",
    alpha_max: float = 0.2,
    alpha_min: float = 0.05,
    start_step: int = 0,
    end_step: int = 1_000_000
) -> float:
    """
    动作增量缩放因子动态调整 - 直接修改ActionTerm实例

    实现动作增量因子的动态调整，支持从较大的探索值线性递减到较小的精细控制值。
    这解决了固定缩放因子导致的前期探索不足和后期精细控制欠佳的问题。

    注意：这个函数直接修改ActionManager中的ActionTerm实例，而不是通过modify_term_cfg。

    Args:
        env: 环境实例
        env_ids: 环境ID列表（未使用，保持接口一致性）
        action_term_name: 动作项名称，默认为"hand_joint_pos"
        alpha_max: 起始缩放因子（较大值，利于前期探索）
        alpha_min: 终止缩放因子（较小值，利于后期精细控制）
        start_step: 开始递减的步数
        end_step: 递减结束的步数（超过此步数后保持alpha_min）

    Returns:
        新的动作缩放因子值（用于日志记录）

    Note:
        动作缩放因子的计算逻辑:
        1. 在start_step之前: scale = alpha_max
        2. 在end_step之后: scale = alpha_min
        3. 在[start_step, end_step]区间内:
           progress = (current_step - start_step) / (end_step - start_step)
           scale = alpha_max - progress * (alpha_max - alpha_min)
           这是一个线性插值过程，确保scale在区间内平滑过渡
    """
    current_step = env.common_step_counter

    # 参数验证
    if alpha_max <= alpha_min:
        raise ValueError(f"alpha_max ({alpha_max}) must be greater than alpha_min ({alpha_min})")
    if end_step <= start_step:
        raise ValueError(f"end_step ({end_step}) must be greater than start_step ({start_step})")

    # 计算新的缩放因子
    if current_step < start_step:
        # 递减开始前：使用最大值
        new_scale = alpha_max
    elif current_step >= end_step:
        # 递减结束后：使用最小值
        new_scale = alpha_min
    else:
        # 递减过程中：线性插值
        progress = (current_step - start_step) / (end_step - start_step)
        new_scale = alpha_max - progress * (alpha_max - alpha_min)

    # 直接修改ActionTerm实例的_scale属性
    try:
        action_term = env.action_manager.get_term(action_term_name)
        if hasattr(action_term, '_scale'):
            old_scale = action_term._scale
            action_term._scale = new_scale
            # print(f"[DEBUG] Step {current_step}: Scale changed from {old_scale} to {new_scale}")
        else:
            print(f"Warning: ActionTerm '{action_term_name}' does not have '_scale' attribute")
            print(f"Available attributes: {[attr for attr in dir(action_term) if not attr.startswith('_')]}")
    except Exception as e:
        print(f"Error modifying action scale: {e}")

    return new_scale


def modify_action_scale_factor_epochs(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    old_value: float,
    alpha_max: float = 0.2,
    alpha_min: float = 0.05,
    start_epoch: int = 0,
    end_epoch: int = 100,
    steps_per_epoch: int = 24000  # 默认：100环境 × 240步长
) -> float:
    """
    基于训练轮次的动作增量缩放因子动态调整

    提供基于epoch的调整接口，内部转换为步数进行计算。
    这种方式更符合强化学习训练的常见表述习惯。

    Args:
        env: 环境实例
        env_ids: 环境ID列表（未使用，保持接口一致性）
        old_value: 当前的scale值
        alpha_max: 起始缩放因子（较大值，利于前期探索）
        alpha_min: 终止缩放因子（较小值，利于后期精细控制）
        start_epoch: 开始递减的轮次
        end_epoch: 递减结束的轮次
        steps_per_epoch: 每轮次的步数（num_envs × horizon_length）

    Returns:
        新的动作缩放因子值

    Example:
        >>> # 配置示例：从第0轮到第100轮，缩放因子从0.2递减到0.05
        >>> action_scale_curriculum = CurrTerm(
        ...     func=mdp.modify_term_cfg,
        ...     params={
        ...         "address": "actions.hand_joint_pos.scale",
        ...         "modify_fn": leaphand_mdp.modify_action_scale_factor_epochs,
        ...         "modify_params": {
        ...             "alpha_max": 0.2,
        ...             "alpha_min": 0.05,
        ...             "start_epoch": 0,
        ...             "end_epoch": 100,
        ...             "steps_per_epoch": 24000  # 100环境 × 240步长
        ...         }
        ...     }
        ... )
    """
    # 转换epoch为步数
    start_step = start_epoch * steps_per_epoch
    end_step = end_epoch * steps_per_epoch

    # 调用基于步数的实现
    return modify_action_scale_factor(
        env=env,
        env_ids=env_ids,
        old_value=old_value,
        alpha_max=alpha_max,
        alpha_min=alpha_min,
        start_step=start_step,
        end_step=end_step
    )





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

