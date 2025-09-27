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

