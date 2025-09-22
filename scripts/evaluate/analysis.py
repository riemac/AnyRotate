# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Analyze reward metrics for a trained RL-Games checkpoint while running inference actions.

- Loads a checkpoint from logs (or a provided path)
- Runs deterministic or stochastic policy inference (not random)
- Collects per-term reward values from Isaac Lab's reward manager every step
- Aggregates and prints detailed statistics (mean, std, percentiles, ranges)
- Optionally saves plots (if matplotlib is available) and JSON summaries

Example:
python scripts/evaluate/analysis.py \
  --task "<YourTaskRegistryPath>" \
  --checkpoint logs/rl_games/<agent_name>/<run_dir>/nn/<agent_name>.pth \
  --epochs 20 --max_steps 3000 --num_envs 8 --deterministic --plot
"""

import argparse
import sys
import json
import math
import os
import time
from typing import Dict, List

import numpy as np
import torch

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Analyze rewards of a trained RL-Games agent during inference.")
parser.add_argument("--task", type=str, required=True, help="Task registry path (same as used for training).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pth) or logs dir.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="If no checkpoint specified, use last saved.")
parser.add_argument("--episodes", type=int, default=1, help="Per-env episodes to run before stopping (all envs must complete this many episodes).")
parser.add_argument("--max_steps", type=int, default=3000, help="Max simulation steps (safety cap).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
parser.add_argument("--plot", action="store_true", help="Save plots of reward trends (requires matplotlib).")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save analysis artifacts (PNG/JSON).")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for computing discounted returns.")
parser.add_argument("--min_episode_steps", type=int, default=0, help="Only count episodes with at least this many steps as accepted (per env).")
parser.add_argument("--print_episode_lengths", action="store_true", help="Print per-env accepted episode lengths.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run roughly in real-time if possible.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# Let AppLauncher add its own args (e.g., device, headless, fabric)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always enable cameras only if plotting videos (not needed here)
# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports that require simulator to be up ---
import gymnasium as gym  # noqa: E402

from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg  # noqa: E402
import leaphand.tasks.manager_based.leaphand  # noqa: F401, E402


def _ensure_output_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _percentiles(arr: np.ndarray, ps=(5, 50, 95)):
    if arr.size == 0:
        return {p: float("nan") for p in ps}
    return {p: float(np.percentile(arr, p)) for p in ps}


def _summarize_series(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "p5": float("nan"), "p50": float("nan"), "p95": float("nan")}
    pct = _percentiles(arr, (5, 50, 95))
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p5": pct[5],
        "p50": pct[50],
        "p95": pct[95],
    }


def _read_tb_training_metric(run_dir: str):
    """Read latest training metric from TensorBoard logs in run_dir.

    Preference: rewards/iter > rewards/step > rewards/time. Returns dict {tag, value} or {}.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore
    except Exception:
        return {}

    if not os.path.isdir(run_dir):
        return {}
    # EventAccumulator can take dir path directly
    try:
        ea = EventAccumulator(run_dir)
        ea.Reload()
        tags = ea.Tags() or {}
        scalars = set(tags.get("scalars", [])) if isinstance(tags, dict) else set()
        for tag in ["rewards/iter", "rewards/step", "rewards/time"]:
            if tag in scalars:
                events = ea.Scalars(tag)
                if events:
                    return {"tag": tag, "value": float(events[-1].value)}
    except Exception:
        return {}
    return {}


def main():
    # Parse env and agent cfg similar to play.py
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # Determine checkpoint to use
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    if resume_path is None or not os.path.exists(resume_path):
        print(f"[ERROR] Checkpoint not found: {args_cli.checkpoint}")
        return
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # Create base env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # Convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Keep a handle to underlying IsaacLab env for reward_manager access
    base_env = env.unwrapped

    # Wrap for RL-Games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"].get("env", {}).get("clip_actions", math.inf)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # Register environment in RL-Games registry
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # Configure agent to load checkpoint
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # Create RL-Games runner and agent
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # Determinism
    if args_cli.deterministic:
        agent.is_deterministic = True

    # Reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs.get("obs", obs)

    # Ensure batch mode and init RNN if used
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    # Access reward manager and term metadata
    if not hasattr(base_env, "reward_manager"):
        print("[ERROR] Underlying environment does not expose reward_manager. Cannot analyze rewards.")
        return
    rm = base_env.reward_manager
    term_names = list(rm._term_names)
    term_weights = [cfg.weight for cfg in rm._term_cfgs]
    weight_vec = torch.tensor(term_weights, device=base_env.device, dtype=torch.float32)

    print("\n[INFO] Reward terms & weights:")
    for i, (n, w) in enumerate(zip(term_names, term_weights)):
        print(f"  {i+1:2d}. {n:<30} weight: {w:+.4f}")

    # Determine per-env episode target (simplified semantics): all envs must complete this many episodes
    episodes_per_env_target = int(args_cli.episodes)

    # Prepare analysis containers
    per_term_series: Dict[str, List[float]] = {name: [] for name in term_names}
    total_reward_series: List[float] = []  # mean across envs per step
    episode_returns: List[float] = []      # per finished episode (raw cumulative)
    discounted_returns: List[float] = []   # per finished episode (discounted)
    per_env_return = torch.zeros(base_env.num_envs, device=base_env.device)
    per_env_discounted_return = torch.zeros(base_env.num_envs, device=base_env.device)
    per_env_step_count = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
    # Episode counting: raw (all) vs accepted (>= min-episode-steps)
    per_env_episode_counts_raw = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
    per_env_episode_counts_acc = torch.zeros(base_env.num_envs, device=base_env.device, dtype=torch.long)
    # Episode lengths (steps)
    episode_lengths_steps_acc: List[int] = []  # accepted episodes (>= min steps)
    episode_lengths_steps_raw: List[int] = []  # all finished episodes (raw)
    per_env_episode_lengths_acc: List[List[int]] = [[] for _ in range(base_env.num_envs)]
    # Debug metrics removed in simplified mode
    track_fall_metrics = False
    term_debug_dist_at_done: List[float] = []
    term_debug_dz_at_done: List[float] = []

    dt = base_env.step_dt
    done_episodes_raw = 0
    done_episodes_acc = 0

    # Output dir
    out_dir = args_cli.output_dir or os.path.join(log_dir, "analysis")
    _ensure_output_dir(out_dir)

    print("\n[INFO] Starting analysis loop...")
    start_wall = time.time()
    while simulation_app.is_running():
        # Stopping criteria
        # All envs must have reached at least K ACCEPTED episodes (strict criterion)
        if torch.all(per_env_episode_counts_acc >= episodes_per_env_target):
            break
        # Use Isaac Lab's common step counter for safety cap
        if base_env.common_step_counter >= args_cli.max_steps:
            print("[WARN] Reached max_steps cap. Stopping analysis loop.")
            break
        step_start = time.time()
        with torch.inference_mode():
            # Agent step
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=agent.is_deterministic)
            # Env step
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs.get("obs", obs)

            # Handle RNN state clearing for done envs
            # dones is a boolean tensor of shape (num_envs,)
            done_indices = dones.nonzero(as_tuple=False).flatten()
            if len(done_indices) > 0 and agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, done_indices, :] = 0.0

            # Collect reward terms from reward_manager
            # rm._step_reward shape: [num_envs, num_terms]
            if hasattr(rm, "_step_reward") and rm._step_reward is not None:
                step_terms = rm._step_reward  # torch.Tensor
                # Record per-term mean across envs for time series
                step_means = step_terms.mean(dim=0)
                for i, name in enumerate(term_names):
                    per_term_series[name].append(float(step_means[i].item()))
                # Total reward per env: IMPORTANT — step_terms are already weighted by term weights
                # So the correct total is the sum over terms without applying weights again
                total_step_per_env = step_terms.sum(dim=1)
                total_reward_series.append(float(total_step_per_env.mean().item()))
                # Accumulate episodic return
                per_env_return += total_step_per_env
                # Accumulate discounted return
                gamma_power = args_cli.gamma ** per_env_step_count.float()
                per_env_discounted_return += total_step_per_env * gamma_power
                per_env_step_count += 1

            # Simplified: no debug fall metrics tracking

            # On done envs, finalize episodic returns
            if len(done_indices) > 0:
                # done_indices contains the environment indices that are done
                for idx in done_indices:
                    ep_len = int(per_env_step_count[idx].item())
                    episode_lengths_steps_raw.append(ep_len)
                    # Accept or ignore based on min-episode-steps
                    if ep_len >= args_cli.min_episode_steps:
                        episode_returns.append(float(per_env_return[idx].item()))
                        discounted_returns.append(float(per_env_discounted_return[idx].item()))
                        episode_lengths_steps_acc.append(ep_len)
                        per_env_episode_counts_acc[idx] += 1
                        per_env_episode_lengths_acc[int(idx)].append(ep_len)
                        done_episodes_acc += 1
                    # Reset per-env accumulators
                    per_env_return[idx] = 0.0
                    per_env_discounted_return[idx] = 0.0
                    per_env_step_count[idx] = 0
                # Update raw counts
                per_env_episode_counts_raw[done_indices] += 1
                done_episodes_raw += len(done_indices)

        # Optional real-time pacing
        if args_cli.real_time:
            sleep_time = dt - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Progress log
        current_steps = int(base_env.common_step_counter)
        if current_steps % 100 == 0 and current_steps > 0:
            mean_total = total_reward_series[-1] if len(total_reward_series) > 0 else float("nan")
            min_ep = int(per_env_episode_counts_acc.min().item())
            mean_ep = float(per_env_episode_counts_acc.float().mean().item())
            print(
                f"  step {current_steps:5d} | episodes/env: min={min_ep:2d}, mean={mean_ep:4.1f}/{episodes_per_env_target} | total_reward_mean: {mean_total:+.4f}"
            )

    elapsed = time.time() - start_wall
    print(
        f"\n[INFO] Analysis finished in {elapsed:.1f}s | steps: {int(base_env.common_step_counter)} | episodes (accepted/raw): {done_episodes_acc}/{done_episodes_raw}"
    )

    # --- Summaries ---
    print("\n" + "=" * 100)
    print("REWARD TERM STATISTICS (per-step means over time)")
    print("Note: Values shown are weighted term contributions as provided by the environment's reward manager.")
    print("      The total reward equals the sum of these weighted terms (no extra weighting applied here).")
    print("=" * 100)

    # Calculate column widths for better alignment
    max_name_len = max(len(name) for name in term_names) if term_names else 10
    name_width = max(max_name_len, 20)

    header_cols = ["Term", "Weight", "Mean", "Std", "Min", "P5", "Median", "P95", "Max"]
    col_widths = [name_width, 10, 10, 10, 10, 10, 10, 10, 10]
    num_format_width = 10  # Consistent width for all numeric columns

    # Print header with proper alignment
    header_parts = []
    for i, (col, width) in enumerate(zip(header_cols, col_widths)):
        if i == 0:  # First column (Term)
            header_parts.append(f"{col:<{width}}")
        else:  # Other columns (right-aligned for numbers)
            header_parts.append(f"{col:>{width}}")
    header_line = " | ".join(header_parts)
    print(header_line)
    print("-" * len(header_line))

    table_rows = []
    for i, name in enumerate(term_names):
        stats = _summarize_series(per_term_series[name])
        row = {
            "term": name,
            "weight": float(term_weights[i]),
            **stats,
        }
        table_rows.append(row)

        # Format row with proper alignment (all numbers right-aligned with consistent width)
        formatted_row = (
            f"{name:<{name_width}} | "
            f"{row['weight']:>{num_format_width}.4f} | "
            f"{row['mean']:>{num_format_width}.4f} | "
            f"{row['std']:>{num_format_width}.4f} | "
            f"{row['min']:>{num_format_width}.4f} | "
            f"{row['p5']:>{num_format_width}.4f} | "
            f"{row['p50']:>{num_format_width}.4f} | "
            f"{row['p95']:>{num_format_width}.4f} | "
            f"{row['max']:>{num_format_width}.4f}"
        )
        print(formatted_row)

    print("\n" + "=" * 60)
    print("TOTAL REWARD (per-step mean across envs)")
    print("=" * 60)
    total_stats = _summarize_series(total_reward_series)
    for k, v in total_stats.items():
        print(f"{k.upper():<8}: {v:+12.4f}")

    print("\n" + "=" * 80)
    print("EPISODIC RETURNS (Raw Cumulative)")
    print("Note: No discount factor (γ) applied, raw cumulative rewards")
    print("=" * 80)
    ep_stats = _summarize_series(episode_returns)
    for k, v in ep_stats.items():
        print(f"{k.upper():<8}: {v:+15.4f}")

    print("\n" + "=" * 80)
    print(f"DISCOUNTED EPISODIC RETURNS (γ={args_cli.gamma})")
    print("Note: Discount factor applied to each step reward")
    print("=" * 80)
    disc_stats = _summarize_series(discounted_returns)
    for k, v in disc_stats.items():
        print(f"{k.upper():<8}: {v:+15.4f}")

    # TensorBoard training metric (Option B)
    tb_metric = _read_tb_training_metric(log_dir)
    if tb_metric:
        print("\n" + "=" * 60)
        print("TRAINING METRIC (TensorBoard)")
        print("=" * 60)
        print(f"{tb_metric['tag']}: {tb_metric['value']:+.7f}  (训练计分指标，非环境内奖励)")

    # Additional analysis
    if len(episode_returns) > 0:
        # Compute average episode length from accepted per-episode lengths (per env)
        avg_episode_length = (sum(episode_lengths_steps_acc) / len(episode_lengths_steps_acc)) if len(episode_lengths_steps_acc) > 0 else 0
        print(f"\nADDITIONAL INFO:")
        print(f"Episodes analyzed    : {len(episode_returns)} (accepted)")
        print(f"Episodes (raw)       : {done_episodes_raw}; ignored (<{args_cli.min_episode_steps} steps): {done_episodes_raw - done_episodes_acc}")
        print(f"Total steps          : {int(base_env.common_step_counter)}")
        print(f"Avg episode length   : {avg_episode_length:.1f} steps (per env)")
        if episodes_per_env_target is not None:
            min_ep = int(per_env_episode_counts_acc.min().item())
            max_ep = int(per_env_episode_counts_acc.max().item())
            mean_ep = float(per_env_episode_counts_acc.float().mean().item())
            print(f"Episodes per env     : min={min_ep}, mean={mean_ep:.2f}, max={max_ep} (target={episodes_per_env_target})")
            # For small num_envs, print the full per-env vector for clarity
            if base_env.num_envs <= 32:
                print(f"Episodes per env vec : {per_env_episode_counts_acc.tolist()} (accepted)")
        if args_cli.print_episode_lengths and base_env.num_envs <= 64:
            print("Per-env accepted episode lengths (steps):")
            for env_id, lens in enumerate(per_env_episode_lengths_acc):
                if len(lens) > 0:
                    print(f"  env[{env_id:02d}]: {lens}")
        print(f"Discount factor (γ)  : {args_cli.gamma}")
        if len(discounted_returns) > 0:
            discount_ratio = (sum(discounted_returns) / sum(episode_returns)) if sum(episode_returns) != 0 else 0
            print(f"Discounted/Raw ratio : {discount_ratio:.4f}")

    # Debug: summarize fall metrics at termination
    if track_fall_metrics and len(term_debug_dist_at_done) > 0:
        dist_stats = _summarize_series(term_debug_dist_at_done)
        dz_stats = _summarize_series(term_debug_dz_at_done)
        print("\n[DEBUG] Termination metrics (distance from reset position):")
        print(f"  DIST  -> mean: {dist_stats['mean']:.4f}, std: {dist_stats['std']:.4f}, min: {dist_stats['min']:.4f}, p50: {dist_stats['p50']:.4f}, p95: {dist_stats['p95']:.4f}, max: {dist_stats['max']:.4f}")
        print(f"  |dz|  -> mean: {dz_stats['mean']:.4f}, std: {dz_stats['std']:.4f}, min: {dz_stats['min']:.4f}, p50: {dz_stats['p50']:.4f}, p95: {dz_stats['p95']:.4f}, max: {dz_stats['max']:.4f}")

    # Save JSON summary
    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "episodes_analyzed": int(done_episodes_acc),
        "episodes_raw": int(done_episodes_raw),
        "min_episode_steps": int(args_cli.min_episode_steps),
        "steps": int(base_env.common_step_counter),
        "per_term": table_rows,
        "total_per_step": total_stats,
        "episodic_returns": ep_stats,
        "discounted_returns": disc_stats,
        "gamma": args_cli.gamma,
    }
    if tb_metric:
        summary["tensorboard_training_metric"] = tb_metric
    json_path = os.path.join(out_dir, "reward_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved JSON summary to: {json_path}")

    # Optional plotting
    if args_cli.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            # Total reward trend
            plt.figure(figsize=(10, 4))
            plt.plot(total_reward_series, label="total_reward_mean_per_step")
            plt.xlabel("Step")
            plt.ylabel("Reward")
            plt.title("Total Reward (mean across envs)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            fig1_path = os.path.join(out_dir, "total_reward_trend.png")
            plt.tight_layout()
            plt.savefig(fig1_path)
            plt.close()

            # Per-term trends (grid)
            n_terms = len(term_names)
            cols = 3
            rows = int(math.ceil(n_terms / cols)) if n_terms > 0 else 1
            plt.figure(figsize=(cols * 4, max(1, rows) * 3))
            for i, name in enumerate(term_names):
                ax = plt.subplot(rows, cols, i + 1)
                ax.plot(per_term_series[name])
                ax.set_title(f"{name} (w={term_weights[i]:+.2f})", fontsize=9)
                ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig2_path = os.path.join(out_dir, "per_term_trends.png")
            plt.savefig(fig2_path)
            plt.close()

            print(f"[INFO] Saved plots to: {fig1_path}, {fig2_path}")
        except Exception as e:
            print(f"[WARN] Plotting skipped due to error: {e}")

    # Close env & app
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

