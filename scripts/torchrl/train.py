# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""使用 TorchRL ClipPPOLoss 在 Isaac Lab 环境中训练策略的入口脚本。

本脚本实现了基于 TorchRL 的 PPO (Proximal Policy Optimization) 算法训练流程。
TorchRL 采用组件化设计哲学，提供模块化的 Loss、Collector、Network 等组件，
由用户手动组装训练循环，而非提供统一的 Runner（与 RL-Games/SKRL 不同）。

Algorithm:
    **PPO (Proximal Policy Optimization)**
    
    - Loss Function: ``torchrl.objectives.ppo.ClipPPOLoss``
    - Advantage Estimation: GAE (Generalized Advantage Estimation)
    - Policy Constraint: Clipped importance sampling ratio
    
    数学公式:
        L_PPO = -E_t[min(r_t(θ)·Â_t, clip(r_t(θ), 1-ε, 1+ε)·Â_t)]
        
        其中:
        - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t): 重要性采样比率
        - Â_t: GAE 优势估计
        - ε: clip 阈值 (默认 0.2)

Features:
    - ✅ 对称 Actor-Critic: Actor 和 Critic 使用相同观测
    - ✅ 非对称 Actor-Critic: 通过 ``separate_losses=True`` 支持特权观测
    - ✅ 多环境并行训练
    - ✅ TensorBoard 日志记录
    - ✅ 模型检查点保存与加载

Asymmetric Actor-Critic:
    如需使用非对称 Actor-Critic（Critic 使用特权信息），需在环境配置中定义不同的观测组：
    
    .. code-block:: python
    
        @configclass
        class ObservationsCfg:
            @configclass
            class PolicyCfg(ObsGroup):  # Actor 观测
                joint_pos = ObsTerm(func=mdp.joint_pos_rel)
                # 仅传感器可测量信息
            
            @configclass
            class CriticCfg(ObsGroup):  # Critic 观测
                joint_pos = ObsTerm(func=mdp.joint_pos_rel)
                object_position = ObsTerm(func=mdp.object_position)  # 特权信息
            
            policy: PolicyCfg = PolicyCfg()
            critic: CriticCfg = CriticCfg()
    
    然后在环境注册时配置 ``obs_groups``，并在 Loss 创建时设置 ``separate_losses=True``。

Note:
    本脚本专用于 PPO 算法。如需使用其他算法（如 SAC、TD3、A2C），需要：
    
    1. 使用对应的 Loss 模块（``SACLoss``, ``TD3Loss``, ``A2CLoss`` 等）
    2. 调整数据收集器（off-policy 算法需要 Replay Buffer）
    3. 修改训练循环逻辑
    
    TorchRL 的组件化设计意味着不同算法需要不同的脚本实现。

Example:
    训练 Cartpole 环境::
    
        $ python train.py --task Isaac-Cartpole-v0 --num_envs 512 --total_frames 1000000
    
    使用检查点恢复训练::
    
        $ python train.py --task Isaac-Cartpole-v0 --checkpoint /path/to/checkpoint.pt

"""

# Launch Isaac Sim Simulator first.

import argparse
import contextlib
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with TorchRL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="torchrl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--total_frames", type=int, default=None, help="Total environment frames to collect for PPO.")
parser.add_argument("--log_interval", type=int, default=10, help="How often (in updates) to log scalars.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a TorchRL checkpoint to resume from.")
parser.add_argument(
    "--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors for manager envs."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import omni
from tensordict import TensorDictBase
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.utils import ValueEstimators

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.torchrl import is_torchrl_available, make_torchrl_env

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import leaphand.tasks  # noqa: F401

# Relative imports must occur after package initialisation
from common import build_actor, build_critic, flatten_size, prepare_optimizer, select_spec, split_key

# PLACEHOLDER: Extension template (do not remove this comment)

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """主训练函数，使用 TorchRL 的 PPO 算法训练强化学习策略。
    
    Args:
        env_cfg: 环境配置对象，支持 ManagerBasedRLEnvCfg、DirectRLEnvCfg 或 DirectMARLEnvCfg
        agent_cfg: 智能体配置字典，包含网络结构、优化器、PPO 超参数等
    
    Raises:
        ImportError: 如果 TorchRL 未安装
    """
    # 检查 TorchRL 是否可用
    if not is_torchrl_available():
        raise ImportError(
            "TorchRL 未安装。请运行 `pip install isaaclab_rl[torchrl]` 或确保当前虚拟环境包含 torchrl。"
        )

    agent_cfg = agent_cfg or {}

    # 使用命令行参数覆盖配置文件中的设置
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 设置随机种子以确保可复现性
    seed = agent_cfg.get("seed", 0)
    if args_cli.seed is not None:
        seed = args_cli.seed
    agent_cfg["seed"] = seed
    env_cfg.seed = seed

    # 确定训练使用的设备（CPU 或 GPU）
    device = torch.device(agent_cfg.get("device", env_cfg.sim.device))

    # 创建日志目录，使用时间戳命名
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "torchrl", args_cli.task or "torchrl"))
    log_dir = os.path.join(log_root_path, run_info)
    os.makedirs(log_dir, exist_ok=True)

    # 设置环境的日志目录
    env_cfg.log_dir = log_dir

    # 如果请求，导出 IO 描述符（仅 ManagerBasedRLEnv 支持）
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        if args_cli.export_io_descriptors:
            omni.log.warn("只有 ManagerBasedRLEnv 支持导出 IO descriptors。")  # type: ignore[attr-defined]

    # 保存配置文件以便后续复现实验
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 保存启动命令
    command = " ".join(sys.orig_argv)
    (Path(log_dir) / "command.txt").write_text(command)

    # 创建 Isaac Lab 环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 如果是多智能体环境，转换为单智能体环境
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)  # type: ignore[arg-type]

    # 如果启用视频录制，添加录制包装器
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 将 Gymnasium 环境转换为 TorchRL 环境
    torchrl_env = make_torchrl_env(env, device=device)  # type: ignore[arg-type]
    torchrl_env.set_seed(seed)

    # 获取观测和动作的键名（支持嵌套字典）
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

    # 获取观测和动作的规格说明
    policy_obs_spec = select_spec(torchrl_env.observation_spec, policy_obs_key)
    critic_obs_spec = select_spec(torchrl_env.observation_spec, critic_obs_key) if has_asymmetric_obs else policy_obs_spec
    action_spec = select_spec(torchrl_env.action_spec, action_key)
    policy_obs_dim = flatten_size(policy_obs_spec.shape)
    critic_obs_dim = flatten_size(critic_obs_spec.shape)

    # 构建策略网络（Actor）和价值网络（Critic）
    policy_cfg = agent_cfg.get("policy_model", {})
    value_cfg = agent_cfg.get("value_model", {})
    policy_module = build_actor(policy_obs_key, action_key, action_spec, policy_obs_dim, policy_cfg, device)
    value_module = build_critic(critic_obs_key, critic_obs_dim, value_cfg, device)

    # 获取 PPO 算法的超参数
    loss_cfg = agent_cfg.get("ppo", {})
    clip_epsilon = loss_cfg.get("clip_epsilon", 0.2)  # PPO 的裁剪参数
    entropy_coef = loss_cfg.get("entropy_coef", 0.0)  # 熵正则化系数
    value_loss_coef = loss_cfg.get("value_loss_coef", 0.5)  # 价值损失系数
    normalize_advantage = loss_cfg.get("normalize_advantage", True)  # 是否归一化优势函数

    # 创建 PPO 损失模块
    loss_module = ClipPPOLoss(
        policy_module,
        value_module,
        clip_epsilon=clip_epsilon,
        critic_coeff=value_loss_coef,
        entropy_bonus=entropy_coef != 0.0,
        entropy_coeff=entropy_coef if entropy_coef != 0.0 else None,
        normalize_advantage=normalize_advantage,
    )
    loss_module.set_keys(action=action_key, value="state_value")
    loss_module.default_value_estimator = ValueEstimators.GAE
    # 配置 GAE (Generalized Advantage Estimation) 优势估计器
    loss_module.make_value_estimator(
        value_type=ValueEstimators.GAE,
        gamma=loss_cfg.get("gamma", 0.99),  # 折扣因子
        lmbda=loss_cfg.get("gae_lambda", 0.95),  # GAE lambda 参数
    )
    # 手动设置一些内部参数（TorchRL 的实现细节）
    loss_module._critic_coef = value_loss_coef
    loss_module.entropy_bonus = entropy_coef != 0.0
    loss_module._entropy_coef = entropy_coef
    loss_module.to(device)

    # 准备优化器（通常是 Adam）
    optimizer = prepare_optimizer(loss_module.parameters(), agent_cfg.get("optimizer", {}))

    # 配置数据收集器
    collector_cfg = agent_cfg.get("collector", {})
    frames_per_batch = collector_cfg.get("frames_per_batch", 16384)  # 每批收集的帧数
    total_frames_cfg = collector_cfg.get("total_frames", int(1e7))  # 总训练帧数
    total_frames = args_cli.total_frames or total_frames_cfg
    init_random_frames = collector_cfg.get("init_random_frames", 0)  # 初始随机探索帧数
    max_frames_per_traj = collector_cfg.get("max_frames_per_traj")
    if max_frames_per_traj in (None, 0):
        max_frames_per_traj = torchrl_env.max_steps or None

    # 创建同步数据收集器
    collector = SyncDataCollector(
        torchrl_env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        init_random_frames=init_random_frames,
        max_frames_per_traj=max_frames_per_traj,
    )
    collector.set_seed(seed)

    # 创建 TensorBoard 日志记录器
    writer = SummaryWriter(os.path.join(log_dir, "tensorboard"))

    # 初始化训练状态变量
    global_frames = 0  # 全局帧计数器
    update_idx = 0  # 更新次数计数器
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 如果提供了检查点路径，加载预训练模型
    if args_cli.checkpoint:
        _load_checkpoint(args_cli.checkpoint, policy_module, value_module, optimizer)

    # 开始训练循环
    with contextlib.ExitStack() as stack:
        # 设置探索类型为随机探索
        stack.enter_context(set_exploration_type(ExplorationType.RANDOM))
        try:
            # 从收集器迭代获取数据批次
            for data in collector:
                # 将数据移动到指定设备
                data = data.to(device)

                # 展平批次维度和时间维度
                rollout = cast(TensorDictBase, data.flatten(0, 1))
                # 计算优势函数和回报
                loss_module.value_estimator(rollout)

                # 获取小批次大小和训练轮数
                minibatch_size = loss_cfg.get("mini_batch_size", 4096)
                num_epochs = loss_cfg.get("num_epochs", 4)

                # 累积指标用于日志记录
                metrics_accumulator = {}
                # 对每个 epoch 进行训练
                for epoch in range(num_epochs):
                    num_samples = rollout.batch_size[0]
                    # 将 rollout 分成小批次进行训练
                    for start in range(0, num_samples, minibatch_size):
                        length = min(minibatch_size, num_samples - start)
                        if length <= 0:
                            continue
                        batch = cast(TensorDictBase, rollout.narrow(0, start, length))
                        if batch.batch_size[0] == 0:
                            continue
                        # 梯度清零
                        optimizer.zero_grad()
                        # 计算损失
                        losses = loss_module(batch)
                        total_loss = _aggregate_losses(losses)
                        # 反向传播
                        total_loss.backward()
                        # 梯度裁剪（防止梯度爆炸）
                        grad_clip = loss_cfg.get("grad_clip", 1.0)
                        if grad_clip is not None and grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), grad_clip)
                        # 更新参数
                        optimizer.step()
                        # 累积指标
                        _accumulate_metrics(metrics_accumulator, losses)

                # 更新全局帧计数器
                frames_in_batch = int(np.prod(data.batch_size))
                global_frames += frames_in_batch
                update_idx += 1

                # 计算平均奖励
                reward_tensor = data["reward"]
                if reward_tensor.ndim > 2:
                    reward_tensor = reward_tensor.squeeze(-1)
                mean_reward = reward_tensor.sum(dim=0).mean().item()
                # 定期打印训练进度
                if update_idx % args_cli.log_interval == 0:
                    print(
                        f"[TorchRL] update={update_idx} frames={global_frames} mean_reward={mean_reward:.3f}"
                    )

                # 写入指标到 TensorBoard
                _write_metrics(writer, metrics_accumulator, global_frames)
                writer.add_scalar("train/mean_reward", mean_reward, global_frames)

                # 定期保存检查点
                if update_idx % args_cli.log_interval == 0:
                    _save_checkpoint(
                        os.path.join(checkpoint_dir, "latest.pt"),
                        policy_module,
                        value_module,
                        optimizer,
                        global_frames,
                        agent_cfg,
                    )
        except KeyboardInterrupt:
            # 用户中断训练时保存检查点
            print("[INFO] Training interrupted by user. Saving checkpoint...")
            _save_checkpoint(
                os.path.join(checkpoint_dir, "interrupt.pt"),
                policy_module,
                value_module,
                optimizer,
                global_frames,
                agent_cfg,
            )
        finally:
            # 清理资源
            collector.shutdown()
            writer.close()
            torchrl_env.close()
            env.close()


def _aggregate_losses(loss_dict: dict) -> torch.Tensor:
    """聚合多个损失项为单个总损失。
    
    Args:
        loss_dict: 包含各个损失项的字典
        
    Returns:
        聚合后的总损失张量
    """
    loss = torch.zeros(1, device=next(iter(loss_dict.values())).device)
    for key in ("loss_objective", "loss_entropy", "loss_critic"):
        if key in loss_dict:
            loss = loss + loss_dict[key]
    return loss


def _accumulate_metrics(container: dict, loss_dict: dict):
    """累积训练指标用于后续日志记录。
    
    Args:
        container: 用于存储累积指标的字典
        loss_dict: 当前批次的损失字典
    """
    for key, value in loss_dict.items():
        if key.startswith("loss") or key in {"kl_approx", "entropy"}:
            container.setdefault(key, []).append(value.detach().cpu())


def _write_metrics(writer: SummaryWriter, metrics: dict, step: int):
    """将累积的指标写入 TensorBoard。
    
    Args:
        writer: TensorBoard SummaryWriter 对象
        metrics: 累积的指标字典
        step: 当前训练步数
    """
    for key, values in metrics.items():
        stacked = torch.stack(values)
        writer.add_scalar(f"loss/{key}", stacked.mean().item(), step)


def _load_checkpoint(path: str, policy, value, optimizer):
    """从检查点文件加载模型和优化器状态。
    
    Args:
        path: 检查点文件路径
        policy: 策略网络模块
        value: 价值网络模块
        optimizer: 优化器对象
    """
    checkpoint = torch.load(path, map_location=policy.device)
    policy.load_state_dict(checkpoint["policy"])
    value.load_state_dict(checkpoint["value"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"[INFO] Loaded checkpoint from {path}")


def _save_checkpoint(path: str, policy, value, optimizer, frames: int, agent_cfg: dict):
    """保存模型和优化器状态到检查点文件。
    
    Args:
        path: 检查点保存路径
        policy: 策略网络模块
        value: 价值网络模块
        optimizer: 优化器对象
        frames: 当前训练的总帧数
        agent_cfg: 智能体配置字典
    """
    torch.save(
        {
            "policy": policy.state_dict(),
            "value": value.state_dict(),
            "optimizer": optimizer.state_dict(),
            "frames": frames,
            "agent_cfg": agent_cfg,
        },
        path,
    )


if __name__ == "__main__":
    main()  # type: ignore[misc]
    simulation_app.close()
