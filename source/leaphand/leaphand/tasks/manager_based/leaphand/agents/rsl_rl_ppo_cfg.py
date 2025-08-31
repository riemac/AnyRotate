# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO配置 - LeapHand连续旋转任务"""

from isaaclab.utils import configclass

from isaaclab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class LeaphandContinuousRotPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """LeapHand连续旋转任务的PPO训练配置"""
    
    num_steps_per_env = 24  # 每个环境的步数
    max_iterations = 15000  # 最大迭代次数
    save_interval = 500  # 保存间隔
    experiment_name = "leaphand_continuous_rot"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # 稍微增加熵系数以鼓励探索
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-4,  # 稍微降低学习率以提高稳定性
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class LeaphandContinuousRotPPORunnerCfgPlay(LeaphandContinuousRotPPORunnerCfg):
    """LeapHand连续旋转任务的PPO推理配置"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 推理时的特殊配置
        self.num_steps_per_env = 32
        self.max_iterations = 0  # 不进行训练
        self.save_interval = 0
        
        # 禁用噪声
        self.policy.init_noise_std = 0.0
        
        # 禁用学习
        self.algorithm.learning_rate = 0.0
