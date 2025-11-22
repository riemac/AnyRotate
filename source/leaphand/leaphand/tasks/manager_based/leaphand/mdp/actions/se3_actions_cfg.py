from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from isaaclab.envs.mdp.actions.actions_cfg import (
    RelativeJointPositionActionCfg,
    EMAJointPositionToLimitsActionCfg,
)

from . import (
    dynamic_relative_joint_position_actions as dyn_rel,
    dynamic_ema_joint_position_to_limits_actions as dyn_ema,
)


@configclass
class LinearDecayRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """线性递减的相对关节位置动作配置。

    字段说明：
    - initial_scale_factor: 初始缩放因子（默认 0.5）
    - final_scale_factor: 最终缩放因子（默认 0.1）
    - init_epochs: 开始线性变化的 epoch（含）
    - end_epochs: 结束线性变化的 epoch（含后保持 final）

    Note:
        - 父类的 ``scale`` 字段不会被直接使用；实际生效的是本配置的动态缩放因子。
    """

    class_type: type[ActionTerm] = dyn_rel.LinearDecayRelativeJointPositionAction

    # 动态缩放参数
    initial_scale_factor: float = 0.5
    final_scale_factor: float = 0.1
    init_epochs: int = 0
    end_epochs: int = 100

    # epoch 长度
    horizon_length: int = 32


@configclass
class LinearDecayAlphaEMAJointPositionToLimitsActionCfg(EMAJointPositionToLimitsActionCfg):
    """线性递减的 EMA JointPositionToLimits 动作配置（对 alpha 进行线性递减）。

    字段说明：
    - initial_alpha: 初始 alpha（默认 0.5）
    - final_alpha: 最终 alpha（默认 0.1）
    - init_epochs: 开始线性变化的 epoch（含）
    - end_epochs: 结束线性变化的 epoch（含后保持 final）

    Note:
        - 父类的 ``alpha`` 字段不会被直接使用；实际生效的是本配置的动态 alpha。
    """

    class_type: type[ActionTerm] = dyn_ema.LinearDecayAlphaEMAJointPositionToLimitsAction

    # 动态 alpha 参数
    initial_alpha: float = 0.5
    final_alpha: float = 0.1
    init_epochs: int = 0
    end_epochs: int = 100
    #  epoch 长度
    horizon_length: int = 32


