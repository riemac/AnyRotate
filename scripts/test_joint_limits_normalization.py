#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""测试LeapHand连续旋转任务 - ManagerBasedRLEnv架构"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="测试LeapHand连续旋转任务 - ManagerBasedRLEnv架构")
parser.add_argument("--num_envs", type=int, default=16, help="环境数量")
parser.add_argument("--task", type=str, default="Isaac-Leaphand-ContinuousRot-Manager-v0", help="任务名称")
parser.add_argument("--seed", type=int, default=None, help="随机种子")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab.envs  # noqa: F401
import leaphand.tasks.manager_based.leaphand  # noqa: F401
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    """主函数"""
    # 解析环境配置
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.headless else None)
    
    # 设置随机种子
    if args_cli.seed is not None:
        env.seed(args_cli.seed)
    
    print("[INFO] 环境创建成功!")
    print(f"[INFO] 环境数量: {env.unwrapped.num_envs}")
    print(f"[INFO] 观测空间: {env.observation_space}")
    print(f"[INFO] 动作空间: {env.action_space}")
    
    # 打印环境配置信息
    print("\n" + "="*50)
    print("环境配置信息:")
    print("="*50)
    print_dict(env.unwrapped.cfg.__dict__, nesting=2)
    
    # 重置环境
    obs, _ = env.reset()
    print(f"\n[INFO] 初始观测:")
    for key, value in obs.items():
        print(f"  - {key}: {value.shape}")
    
    # 检查是否支持非对称观测
    if hasattr(env.unwrapped, 'observation_manager'):
        obs_manager = env.unwrapped.observation_manager
        print(f"[INFO] 观测管理器组: {list(obs_manager.group_obs_dim.keys())}")
        for group_name, dim in obs_manager.group_obs_dim.items():
            print(f"  - {group_name}: {dim}")

    # 检查奖励管理器
    if hasattr(env.unwrapped, 'reward_manager'):
        reward_manager = env.unwrapped.reward_manager
        print(f"[INFO] 奖励管理器项: {list(reward_manager.active_terms)}")

    # 检查终止管理器
    if hasattr(env.unwrapped, 'termination_manager'):
        termination_manager = env.unwrapped.termination_manager
        print(f"[INFO] 终止管理器项: {list(termination_manager.active_terms)}")
    
    # 运行几个步骤来测试环境
    print(f"\n[INFO] 开始运行测试...")
    
    for step in range(100):
        # 生成随机动作
        actions = torch.randn(env.unwrapped.num_envs, env.action_space.shape[0], device=env.unwrapped.device)
        actions = torch.clamp(actions, -1.0, 1.0)  # 限制动作范围
        
        # 执行动作
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 打印信息
        if step % 20 == 0:
            print(f"步骤 {step}:")
            print(f"  奖励均值: {rewards.mean().item():.4f}")
            print(f"  奖励标准差: {rewards.std().item():.4f}")
            print(f"  终止环境数: {terminated.sum().item()}")
            print(f"  截断环境数: {truncated.sum().item()}")
            
            # 打印额外信息
            if "log" in info:
                log_info = info["log"]
                print("  奖励分解:")
                for key, value in log_info.items():
                    if "reward" in key.lower():
                        print(f"    {key}: {value:.4f}")
    
    print(f"\n[INFO] 测试完成!")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
