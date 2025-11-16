# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务环境配置 - ManagerBasedRLEnv架构

"""

from isaaclab.utils import configclass

import inhand_base_env_cfg

# from .mdp.actions import LinearDecayAlphaEMAJointPositionToLimitsActionCfg

@configclass
class ActionsCfg:
    """动作配置 - 动作平滑"""
    hand_joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],  # 所有手部关节
        scale=1.0,  # 动作缩放因子（对EMA类型影响不大，因为有rescale_to_limits）
        rescale_to_limits=True,  # 将[-1,1]动作自动映射到关节限制
        alpha=1/24,  # 平滑系数
    )

@configclass
class InHandIKEnvCfg(inhand_base_env_cfg.InHandObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to leaphand
        self.scene.robot = inhand_base_env_cfg.LEAPHAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # enable clone in fabric
        self.scene.clone_in_fabric = True