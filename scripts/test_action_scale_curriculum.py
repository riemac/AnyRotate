#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
测试动作缩放因子动态调整功能

这个脚本用于验证LeapHand连续旋转任务中的动作增量因子动态调整机制。
测试不同的调整策略（保守型、激进型、基于轮次）的效果。

使用方法:
    python scripts/test_action_scale_curriculum.py --strategy conservative
    python scripts/test_action_scale_curriculum.py --strategy aggressive  
    python scripts/test_action_scale_curriculum.py --strategy default
"""

import argparse
import torch

# 添加命令行参数
parser = argparse.ArgumentParser(description="测试动作缩放因子动态调整功能")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--strategy", type=str, default="default",
                   choices=["default", "conservative", "aggressive"],
                   help="动作缩放调整策略")
parser.add_argument("--test_steps", type=int, default=1000, help="测试步数")

# 启动Isaac Sim应用
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 导入依赖（必须在AppLauncher之后）
from isaaclab.envs import ManagerBasedRLEnv
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import (
    LeaphandContinuousRotActionScaleEnvCfg,
    LeaphandContinuousRotConservativeScaleEnvCfg
)


def test_action_scale_curriculum():
    """测试动作缩放因子动态调整功能"""
    
    # 根据策略选择环境配置
    if args_cli.strategy == "conservative":
        env_cfg = LeaphandContinuousRotConservativeScaleEnvCfg()
        print("🔧 使用保守型动作缩放调整策略")
    elif args_cli.strategy == "aggressive":
        # 暂时使用默认配置，因为激进型配置还未完全实现
        env_cfg = LeaphandContinuousRotActionScaleEnvCfg()
        print("🚀 使用激进型动作缩放调整策略（暂用默认配置）")
    else:
        env_cfg = LeaphandContinuousRotActionScaleEnvCfg()
        print("⚖️ 使用默认动作缩放调整策略")
    
    # 设置环境参数
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 创建环境
    print(f"🌍 创建环境: {args_cli.num_envs} 个并行环境")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 获取动作维度
    action_dim = env.action_manager.total_action_dim
    print(f"🎮 动作维度: {action_dim}")
    
    # 测试循环
    print(f"🧪 开始测试 {args_cli.test_steps} 步...")
    
    # 记录缩放因子变化
    scale_history = []
    step_history = []
    
    # 重置环境
    obs, _ = env.reset()
    
    for step in range(args_cli.test_steps):
        # 生成随机动作
        actions = torch.randn(env.num_envs, action_dim, device=env.device) * 0.5
        
        # 执行动作
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 记录当前的动作缩放因子（每100步记录一次）
        if step % 100 == 0:
            try:
                # 获取当前的动作缩放因子
                action_term = env.action_manager.get_term("hand_joint_pos")
                if hasattr(action_term, '_scale'):
                    current_scale = float(action_term._scale)
                    scale_history.append(current_scale)
                    step_history.append(step)
                    
                    print(f"步数 {step:6d}: 动作缩放因子 = {current_scale:.6f}, "
                          f"平均奖励 = {rewards.mean().item():.4f}")
            except Exception as e:
                print(f"⚠️ 无法获取缩放因子: {e}")
    
    # 输出测试结果
    print("\n📊 测试结果总结:")
    print(f"   总测试步数: {args_cli.test_steps}")
    print(f"   并行环境数: {args_cli.num_envs}")
    print(f"   调整策略: {args_cli.strategy}")
    
    if scale_history:
        print(f"   初始缩放因子: {scale_history[0]:.6f}")
        print(f"   最终缩放因子: {scale_history[-1]:.6f}")
        print(f"   缩放因子变化: {scale_history[0] - scale_history[-1]:.6f}")
        
        # 检查是否按预期递减
        if len(scale_history) > 1:
            is_decreasing = all(scale_history[i] >= scale_history[i+1] 
                              for i in range(len(scale_history)-1))
            print(f"   是否按预期递减: {'✅ 是' if is_decreasing else '❌ 否'}")
    
    # 关闭环境
    env.close()
    print("✅ 测试完成!")


def main():
    """主函数"""
    try:
        test_action_scale_curriculum()
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断测试")
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭仿真应用
        simulation_app.close()


if __name__ == "__main__":
    main()
