# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""载入 TorchRL PPO 训练得到的策略并在 Isaac Lab 环境中回放。

本脚本用于评估通过 TorchRL PPO 训练得到的策略性能，支持：

- 从检查点加载训练好的 Actor 网络
- 在仿真环境中可视化策略行为
- 记录回放视频用于分析
- 统计平均奖励等评估指标

Algorithm:
    本脚本专用于 **PPO** 算法训练的模型。如果使用其他算法（SAC、TD3 等）训练，
    需要修改 Actor 构建逻辑以匹配对应算法的网络结构。

Note:
    - 回放时模型处于确定性模式（``set_exploration_type(ExplorationType.MODE)``）
    - 仅加载 Actor 网络，不需要 Critic
    - 检查点必须由 ``train.py`` 脚本生成

Example:
    使用最新检查点回放::
    
        $ python play.py --task Isaac-Cartpole-v0 --use_last_checkpoint --num_envs 32
    
    使用指定检查点并录制视频::
    
        $ python play.py --task Isaac-Cartpole-v0 --checkpoint /path/to/model.pt --video

"""

# Launch Isaac Sim Simulator first.

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a TorchRL PPO policy in Isaac Lab.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=1000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a saved TorchRL checkpoint (.pt).")
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When checkpoint is not provided, pick the most recent checkpoint from logs/torchrl/<task>.",
)
parser.add_argument("--max_steps", type=int, default=2000, help="Maximum number of simulation steps to run.")
parser.add_argument("--seed", type=int, default=None, help="Environment random seed.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from typing import cast

from tensordict import TensorDictBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.torchrl import is_torchrl_available, make_torchrl_env

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path
import leaphand.tasks  # noqa: F401

from common import build_actor, build_critic, flatten_size, select_spec, split_key

# PLACEHOLDER: Extension template (do not remove this comment)

# TorchRL 配置入口点（与环境注册中的键名匹配）
agent_cfg_entry_point = "torchrl_cfg_entry_point"

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    if not is_torchrl_available():
        raise ImportError("TorchRL 未安装，无法回放策略。请先安装 isaaclab_rl[torchrl] 依赖。")

    agent_cfg = agent_cfg or {}

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    seed = agent_cfg.get("seed", 0)
    if args_cli.seed is not None:
        seed = args_cli.seed
    agent_cfg["seed"] = seed
    env_cfg.seed = seed

    device = torch.device(agent_cfg.get("device", env_cfg.sim.device))

    task_name = args_cli.task.split(":")[-1]
    log_root_path = os.path.abspath(os.path.join("logs", "torchrl", task_name))

    if args_cli.checkpoint is not None:
        checkpoint_path = args_cli.checkpoint
    else:
        if not args_cli.use_last_checkpoint:
            raise ValueError("请通过 --checkpoint 指定模型路径，或添加 --use_last_checkpoint 自动搜索。")
        checkpoint_path = get_checkpoint_path(os.path.join(log_root_path, ""), ".*", "latest.pt", sort_alpha=False)

    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"[INFO] Loading TorchRL checkpoint: {checkpoint_path}")

    env_cfg.log_dir = os.path.dirname(checkpoint_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)  # type: ignore[arg-type]

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(env_cfg.log_dir, "videos", "play"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    torchrl_env = make_torchrl_env(env, device=device)  # type: ignore[arg-type]
    torchrl_env.set_seed(seed)

    # 检测是否为非对称 Actor-Critic（有 policy/critic 观测组）
    obs_spec_keys = list(torchrl_env.observation_spec.keys())
    has_asymmetric_obs = "policy" in obs_spec_keys and "critic" in obs_spec_keys
    
    if has_asymmetric_obs:
        # 非对称观测：Actor 使用 policy 观测，Critic 使用 critic 观测
        policy_obs_key = split_key(agent_cfg.get("policy_obs_key", "policy"))
        critic_obs_key = split_key(agent_cfg.get("critic_obs_key", "critic"))
        obs_key = policy_obs_key  # Actor 默认使用 policy 观测
    else:
        # 对称观测：Actor 和 Critic 使用相同观测
        obs_key = split_key(agent_cfg.get("obs_key", "observation"))
        policy_obs_key = critic_obs_key = obs_key
    
    action_key = split_key(agent_cfg.get("action_key", "action"))

    policy_obs_spec = select_spec(torchrl_env.observation_spec, policy_obs_key)
    critic_obs_spec = select_spec(torchrl_env.observation_spec, critic_obs_key) if has_asymmetric_obs else policy_obs_spec
    
    # 对于 action_spec，如果不是复合类型（即没有嵌套结构），直接使用
    if hasattr(torchrl_env.action_spec, 'keys') and len(action_key) > 0 and action_key[0] in torchrl_env.action_spec.keys():
        action_spec = select_spec(torchrl_env.action_spec, action_key)
    else:
        action_spec = torchrl_env.action_spec
    
    # 计算观测维度（跳过 batch 维度，即第一个维度）
    policy_obs_dim = flatten_size(policy_obs_spec.shape[1:])
    critic_obs_dim = flatten_size(critic_obs_spec.shape[1:])

    policy_cfg = agent_cfg.get("policy_model", {})
    value_cfg = agent_cfg.get("value_model", {})

    policy_module = build_actor(policy_obs_key, action_key, action_spec, policy_obs_dim, policy_cfg, device)
    value_module = build_critic(critic_obs_key, critic_obs_dim, value_cfg, device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_module.load_state_dict(checkpoint["policy"])
    if "value" in checkpoint:
        value_module.load_state_dict(checkpoint["value"])

    torchrl_env.eval()
    policy_module.eval()

    total_reward = torch.zeros(torchrl_env.batch_size, device=device)
    step_count = 0

    td = cast(TensorDictBase, torchrl_env.reset())
    with set_exploration_type(ExplorationType.MODE):
        while simulation_app.is_running():
            if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
                break
            with torch.inference_mode():
                td = policy_module(td)
                td = torchrl_env.step(td)

            reward = td.get("reward").squeeze(-1)
            total_reward += reward
            step_count += 1

            if td.get("done").any():
                print(
                    f"[INFO] Episode finished at step {step_count} | mean reward: {total_reward.mean().item():.3f}"
                )
                break

            td = cast(TensorDictBase, td.get("next"))

    torchrl_env.close()
    env.close()


if __name__ == "__main__":
    main()  # type: ignore[misc]
    simulation_app.close()
