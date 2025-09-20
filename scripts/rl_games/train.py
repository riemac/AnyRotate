# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# ======= 命令行参数定义区 =======
# 使用 argparse 定义脚本可接受的命令行参数，便于在终端灵活配置训练行为
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
# 将 AppLauncher 支持的命令行参数追加到 parser 中（例如：render、headless 等）
AppLauncher.add_app_launcher_args(parser)
# 解析已知参数，并保留其余参数供 Hydra 使用（hydra_args）
args_cli, hydra_args = parser.parse_known_args()
# 如果启用录像，则必须启用相机才能采集图像
if args_cli.video:
    args_cli.enable_cameras = True

# 为了让 Hydra 正确接收剩余参数，清空 sys.argv，并保留脚本名与 hydra_args
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Omniverse/Isaac Sim 应用（通过 AppLauncher 初始化）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import leaphand.tasks.manager_based.leaphand  # noqa: F401

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    # ======= 使用非-Hydra 的 CLI 参数覆盖 Hydra 配置（优先级：命令行 > 配置文件） =======
    # 调整并行环境数量（若用户在命令行指定）
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # 强制指定仿真设备（例如 cpu / cuda:0 / cuda:1）
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 如果用户指定 seed 为 -1，则随机采样一个 seed。注意这里 -1 被视为“随机”占位符
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # 将 seed 写入 agent 配置，供训练与环境复现使用
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # 将最大训练迭代次数写入 agent 配置（覆盖配置文件中的值）
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    # 如果指定了 checkpoint，则解析路径并设置加载标志与路径
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    # 如果命令行指定了 sigma，则将其转换为 float 并传递给 RL-Games runner
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # ======= 多卡/分布式训练设置 =======
    if args_cli.distributed:
        # 在多卡训练场景中，使得每个进程有不同的随机种子以避免完全相同的初始化
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        # 将 agent 的训练设备设置成本地可见的 GPU（local_rank）
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # 同步更新环境仿真设备（否则环境可能仍使用默认设备）
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # 在环境初始化之前设置 env_cfg.seed（某些随机化在环境构造时发生）
    env_cfg.seed = agent_cfg["params"]["seed"]

    # ======= 日志与实验目录设置 =======
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.join("logs", "rl_games", config_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # 如果没有指定完整实验名称，则使用当前时间戳作为标识
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # 将日志目录及完整实验名称保存回 agent 配置，便于 RL-Games 内部与外部工具使用
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
    experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name

    # 将 env 和 agent 配置持久化到磁盘，便于复现与调试（YAML + pickle）
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # ======= 从 agent_cfg 读取训练相关设置 =======
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # ======= 创建 Isaac 环境（通过 Gym 接口） =======
    # 传入 hydra 解析后的 env_cfg 对象供环境构造使用；若要录像，则 render_mode 为 rgb_array
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 如果环境为多智能体 DirectMARLEnv，而所用算法只支持单智能体，则转换为单智能体接口
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ======= 视频录制封装（可选） =======
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            # 触发录制的条件函数：step % video_interval == 0 时开始录制
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        # 使用 Gym 的 RecordVideo wrapper 将环境包装以保存训练视频
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ======= 将环境包装成 RL-Games 能够使用的向量化环境 ======= # IDEA: 如果集成第三方RL库，这个可能是要改动的部分
    # RlGamesVecEnvWrapper 负责将单环境或多环境适配成 RL-Games 所需的接口
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # ======= 向 RL-Games 注册自定义 VecEnv 和环境实例 =======
    # 注册一个名为 "IsaacRlgWrapper" 的 vecenv 工厂函数，返回 RlGamesGpuEnv 实例
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    # 将一个名为 "rlgpu" 的环境配置注册到 rl-games 的 env_configurations 中
    # 这样在 agent 配置中使用 env: rlgpu 时，会创建上面注册的 vecenv
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # 将实际并行 actor 数量写入 agent 配置，供 RL-Games 运行器使用
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # 使用 IsaacAlgoObserver 创建 runner（该 observer 为 rl-games 提供 Isaac/Sim 的统计回调）
    runner = Runner(IsaacAlgoObserver())
    # 加载 agent 配置（包含算法、网络与训练超参等）
    runner.load(agent_cfg)

    # 重置 agent 与环境的内部状态（准备训练）
    runner.reset()
    # ======= 训练或评估启动 =======
    # 在分布式场景下，仅在全局 rank 为 0 的进程初始化 Weights & Biases（wandb）监控
    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        # 初始化 wandb（可选），并将配置上传以便可视化与对比
        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        # 将 env/agent 配置写入 wandb config 中，便于线上查看与复现
        wandb.config.update({"env_cfg": env_cfg.to_dict()})
        wandb.config.update({"agent_cfg": agent_cfg})

    # 如果指定 checkpoint，则在 runner.run 中传入 checkpoint 路径以加载模型后继续训练/评估
    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # 训练完成后关闭环境（释放资源）
    env.close()


if __name__ == "__main__":
    # 执行主函数（主逻辑已由 hydra_task_config 装饰器封装，hydra 会处理配置注入）
    main()
    # 关闭仿真应用（确保 Omniverse/IsaacSim 进程被正确终止）
    simulation_app.close()
