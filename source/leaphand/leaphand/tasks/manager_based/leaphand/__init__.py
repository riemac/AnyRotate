# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务 - ManagerBasedRLEnv架构"""

import gymnasium as gym

from . import agents
from .leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg
from .inhand_env_cfg import InHandEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Template-Leaphand-ContinuousRot-Manager-v0", # Template可被list_envs.py识别（但不影响环境注册与训练）
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leaphand_continuous_rot_env_cfg:LeaphandContinuousRotEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeaphandContinuousRotPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Template-InHand-Manager-v0", # [项目名]-[任务名]-[架构名]-v[版本号]
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inhand_env_cfg:InHandEnvCfg", # 包路径.模块名:类名
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_inhand_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeaphandContinuousRotPPORunnerCfgPlay",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)