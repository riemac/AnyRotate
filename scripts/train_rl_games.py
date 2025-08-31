#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用rl_games训练LeapHand连续旋转任务"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="使用rl_games训练LeapHand连续旋转任务")
parser.add_argument("--video", action="store_true", default=False, help="录制训练视频")
parser.add_argument("--video_length", type=int, default=200, help="视频长度")
parser.add_argument("--video_interval", type=int, default=2000, help="视频录制间隔")
parser.add_argument("--num_envs", type=int, default=None, help="环境数量")
parser.add_argument(
    "--task", type=str, default="Isaac-Leaphand-ContinuousRot-Manager-v0", help="任务名称"
)
parser.add_argument("--seed", type=int, default=None, help="随机种子")
parser.add_argument("--max_iterations", type=int, default=None, help="最大训练迭代次数")
parser.add_argument("--checkpoint", type=str, default=None, help="检查点路径")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime

from isaaclab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import isaaclab.envs  # noqa: F401
import leaphand.tasks.manager_based.leaphand  # noqa: F401


def main():
    """主函数"""
    # 解析环境配置
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    
    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # 包装环境以适配rl_games
    env = RlGamesVecEnvWrapper(env)
    
    print("[INFO] 环境创建成功!")
    print(f"[INFO] 环境数量: {env.num_envs}")
    print(f"[INFO] 观测空间: {env.observation_space}")
    print(f"[INFO] 动作空间: {env.action_space}")
    
    # 设置rl_games配置
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import torch_ext
    
    # 注册环境
    vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: env,
    })
    
    # 加载rl_games配置
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / "source/leaphand/leaphand/tasks/manager_based/leaphand/agents/rl_games_ppo_cfg.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新配置
    config['params']['config']['num_actors'] = env.num_envs
    if args_cli.seed is not None:
        config['params']['seed'] = args_cli.seed
    if args_cli.max_iterations is not None:
        config['params']['config']['max_epochs'] = args_cli.max_iterations
    
    # 设置检查点路径
    if args_cli.checkpoint is not None:
        config['params']['load_checkpoint'] = True
        config['params']['load_path'] = args_cli.checkpoint
    
    # 设置实验名称和日志目录
    experiment_name = f"leaphand_continuous_rot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config['params']['config']['name'] = experiment_name
    
    # 创建日志目录
    log_dir = f"logs/rl_games/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"[INFO] 开始训练，实验名称: {experiment_name}")
    print(f"[INFO] 日志目录: {log_dir}")
    print(f"[INFO] 最大迭代次数: {config['params']['config']['max_epochs']}")
    
    # 创建并运行训练器
    runner = Runner()
    runner.load(config)
    runner.reset()
    runner.run({
        'train': True,
        'play': False,
        'checkpoint': args_cli.checkpoint,
        'sigma': None
    })
    
    print("[INFO] 训练完成!")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
