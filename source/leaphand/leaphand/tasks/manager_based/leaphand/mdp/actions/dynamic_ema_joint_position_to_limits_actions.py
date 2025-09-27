from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.joint_actions_to_limits import EMAJointPositionToLimitsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .dynamic_actions_cfg import (
        LinearDecayAlphaEMAJointPositionToLimitsActionCfg,
    )


class AbstractDynamicAlphaEMAJointPositionToLimitsAction(EMAJointPositionToLimitsAction, ABC):
    """EMA 关节位置到极限的动态 alpha 抽象基类。

    仅在 epoch 边界调用 :meth:`update_scale_factor` 更新 alpha；在一个
    epoch（horizon_length 个 step）内保持不变。

    Args:
        cfg: 子类对应的配置对象，需至少包含 ``initial_alpha`` 字段。
        env: 管理器式环境实例。

    Note:
        - 对 EMA 而言，真正的“缩放因子”是 ``alpha``（权重），非 ``scale``。
        - 本类会将 ``self._alpha`` 同步到 ``current_alpha``。
    """

    _current_alpha: float

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        init = float(getattr(cfg, "initial_alpha", 1.0))
        self._current_alpha = init
        # 覆写父类解析得到的 alpha（可能来自 cfg.alpha），以动态 alpha 为准
        self._alpha = self._current_alpha

        # 读取 horizon_length（必须 >0）
        self._horizon_length = int(getattr(cfg, "horizon_length", 32))
        if self._horizon_length <= 0:
            raise ValueError("horizon_length 必须为正整数")
        # 记录上次已更新的 epoch（初始为 -1，确保首步会更新）
        self._last_epoch = -1
        # 选择计数来源：优先环境计数器，否则内部计数器
        self._use_env_counter = hasattr(self._env, "common_step_counter")
        self._internal_step_counter = 0

    @property
    def current_alpha(self) -> float:
        """当前生效的 EMA alpha。"""
        return self._current_alpha

    def _set_alpha(self, value: float) -> None:
        """设置当前 alpha 并同步到内部 ``_alpha``。

        Args:
            value: 新的 alpha（范围建议在 [0, 1]）。
        """
        # 限制在 [0, 1]
        v = max(0.0, min(1.0, float(value)))
        self._current_alpha = v
        self._alpha = v

    @abstractmethod
    def update_scale_factor(self, current_epoch: int) -> None:
        """在 epoch 边界更新 alpha。

        Args:
            current_epoch: 当前 epoch 索引（从 0 开始）。
        """
        raise NotImplementedError


    def process_actions(self, actions):
        """
        在每个环境步自动检查并更新 alpha，然后执行标准 EMA 预处理。

        Args:
            actions: 输入动作张量，形状 (num_envs, action_dim)。
        """
        # 1) 计算当前 step 与 epoch（优先使用环境计数器）
        step_count = getattr(self._env, "common_step_counter", None) if self._use_env_counter else None
        if step_count is None:
            step_count = self._internal_step_counter
        epoch = step_count // self._horizon_length
        # 2) 在 epoch 边界触发 alpha 更新
        if epoch != self._last_epoch:
            self.update_scale_factor(epoch)
            self._last_epoch = epoch
        # 3) 执行父类的预处理（会使用更新后的 self._alpha）
        super().process_actions(actions)
        # 4) 若使用内部计数器，则步进一次
        if not self._use_env_counter:
            self._internal_step_counter += 1

class LinearDecayAlphaEMAJointPositionToLimitsAction(AbstractDynamicAlphaEMAJointPositionToLimitsAction):
    """线性递减的 EMA JointPositionToLimits 动作（对 alpha 进行线性递减）。

    线性递减公式：
    ``alpha = initial - (initial - final) * (current_epoch - init_epochs) / (end_epochs - init_epochs)``。

    Args:
        cfg: :class:`LinearDecayAlphaEMAJointPositionToLimitsActionCfg` 配置对象。
        env: 管理器式环境实例。

    Note:
        - 当 ``current_epoch < init_epochs`` 时使用 ``initial_alpha``。
        - 当 ``current_epoch >= end_epochs`` 时使用 ``final_alpha``。
        - 仅在 epoch 结束时调用 ``update_scale_factor``。
    """

    def __init__(self, cfg: LinearDecayAlphaEMAJointPositionToLimitsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.end_epochs < cfg.init_epochs:
            raise ValueError("end_epochs 必须 >= init_epochs")
        self._initial = float(cfg.initial_alpha)
        self._final = float(cfg.final_alpha)
        self._init_epochs = int(cfg.init_epochs)
        self._end_epochs = int(cfg.end_epochs)
        # 确保初始生效
        self._set_alpha(self._initial)

    def update_scale_factor(self, current_epoch: int) -> None:
        if current_epoch < self._init_epochs:
            value = self._initial
        elif current_epoch >= self._end_epochs or self._end_epochs == self._init_epochs:
            value = self._final
        else:
            ratio = (current_epoch - self._init_epochs) / float(self._end_epochs - self._init_epochs)
            value = self._initial - (self._initial - self._final) * ratio
        self._set_alpha(value)

