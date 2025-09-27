from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .dynamic_actions_cfg import (
        LinearDecayRelativeJointPositionActionCfg,
    )


class AbstractDynamicScaleRelativeJointPositionAction(RelativeJointPositionAction, ABC):
    """相对关节位置控制的动态缩放抽象基类。

    仅在 epoch 边界调用 :meth:`update_scale_factor` 更新缩放因子；在一个
    epoch（horizon_length 个 step）内保持不变。

    Args:
        cfg: 子类对应的配置对象，需至少包含 ``initial_scale_factor`` 字段。
        env: 管理器式环境实例。

    Note:
        - 动态缩放因子用于替代 JointAction 中的 ``scale``，本类会将
          ``self._scale`` 同步到 ``current_scale_factor``。
        - 建议在训练循环每个 epoch 结束时调用 ``update_scale_factor(current_epoch)``。
    """

    _current_scale_factor: float

    def __init__(self, cfg, env: ManagerBasedEnv):  # cfg 由具体子类的 Cfg 类型约束
        super().__init__(cfg, env)
        # 初始化当前缩放因子
        init = float(getattr(cfg, "initial_scale_factor", 1.0))
        self._current_scale_factor = init
        # 覆写父类解析得到的 scale（如果有），使用动态缩放因子
        self._scale = self._current_scale_factor

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
    def current_scale_factor(self) -> float:
        """当前生效的缩放因子。"""
        return self._current_scale_factor

    def _set_scale_factor(self, value: float) -> None:
        """设置当前缩放因子并同步到内部 ``_scale``。

        Args:
            value: 新的缩放因子（建议非负）。
        """
        self._current_scale_factor = float(value)
        self._scale = self._current_scale_factor

    @abstractmethod
    def update_scale_factor(self, current_epoch: int) -> None:
        """在 epoch 边界更新缩放因子。

        Args:
            current_epoch: 当前 epoch 索引（从 0 开始）。
        """
        raise NotImplementedError


    def process_actions(self, actions):
        """
        在每个环境步自动检查并更新缩放因子，然后执行标准动作预处理。

        Args:
            actions: 输入动作张量，形状 (num_envs, action_dim)。
        """
        # 1) 计算当前 step 与 epoch（优先使用环境计数器）
        step_count = getattr(self._env, "common_step_counter", None) if self._use_env_counter else None
        if step_count is None:
            step_count = self._internal_step_counter
        epoch = step_count // self._horizon_length
        # 2) 在 epoch 边界触发缩放因子更新
        if epoch != self._last_epoch:
            self.update_scale_factor(epoch)
            self._last_epoch = epoch
        # 3) 执行父类的预处理（会使用更新后的 self._scale）
        super().process_actions(actions)
        # 4) 若使用内部计数器，则步进一次
        if not self._use_env_counter:
            self._internal_step_counter += 1

class LinearDecayRelativeJointPositionAction(AbstractDynamicScaleRelativeJointPositionAction):
    """线性递减的相对关节位置动作。

    线性递减公式：
    ``scale = initial - (initial - final) * (current_epoch - init_epochs) / (end_epochs - init_epochs)``。

    Args:
        cfg: :class:`LinearDecayRelativeJointPositionActionCfg` 配置对象。
        env: 管理器式环境实例。

    Note:
        - 当 ``current_epoch < init_epochs`` 时使用 ``initial_scale_factor``。
        - 当 ``current_epoch >= end_epochs`` 时使用 ``final_scale_factor``。
        - 仅在 epoch 结束时调用 ``update_scale_factor``。
    """

    def __init__(self, cfg: LinearDecayRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.end_epochs < cfg.init_epochs:
            raise ValueError("end_epochs 必须 >= init_epochs")
        self._initial = float(cfg.initial_scale_factor)
        self._final = float(cfg.final_scale_factor)
        self._init_epochs = int(cfg.init_epochs)
        self._end_epochs = int(cfg.end_epochs)
        # 确保初始生效
        self._set_scale_factor(self._initial)

    def update_scale_factor(self, current_epoch: int) -> None:
        if current_epoch < self._init_epochs:
            value = self._initial
        elif current_epoch >= self._end_epochs or self._end_epochs == self._init_epochs:
            value = self._final
        else:
            ratio = (current_epoch - self._init_epochs) / float(self._end_epochs - self._init_epochs)
            value = self._initial - (self._initial - self._final) * ratio
        self._set_scale_factor(value)

