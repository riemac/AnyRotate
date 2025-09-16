#!/usr/bin/env python3

"""
调试动作缩放问题的脚本
对比DirectRLEnv和ManagerBasedRLEnv在不同环境数量下的动作处理差异
"""

import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="调试动作缩放问题")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 导入必要的模块
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import scale_transform, unscale_transform

# 导入环境配置
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg


def debug_action_scaling():
    """调试动作缩放问题"""
    
    print("=" * 80)
    print(f"调试动作缩放问题 - 环境数量: {args_cli.num_envs}")
    print("=" * 80)
    
    # 创建环境配置
    env_cfg = LeaphandContinuousRotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = 10.0  # 短episode用于测试
    
    # 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 重置环境
    obs, _ = env.reset()
    
    # 获取机器人资产
    robot = env.scene["robot"]
    
    print(f"环境数量: {env.num_envs}")
    print(f"关节数量: {robot.num_joints}")
    print(f"动作维度: {env.action_manager.total_action_dim}")
    
    # 获取关节限制
    joint_limits_lower = robot.data.soft_joint_pos_limits[:, :, 0]  # (num_envs, num_joints)
    joint_limits_upper = robot.data.soft_joint_pos_limits[:, :, 1]  # (num_envs, num_joints)
    
    print(f"\n关节限制范围 (第一个环境):")
    print(f"下限: {joint_limits_lower[0].cpu().numpy()}")
    print(f"上限: {joint_limits_upper[0].cpu().numpy()}")
    
    # 获取动作管理器中的关节动作项
    action_term = env.action_manager._terms["hand_joint_pos"]
    print(f"\n动作项配置:")
    print(f"类型: {type(action_term)}")
    print(f"rescale_to_limits: {action_term.cfg.rescale_to_limits}")
    print(f"scale: {action_term.cfg.scale}")
    
    # 测试不同的动作输入
    test_actions = [
        torch.ones(env.num_envs, env.action_manager.total_action_dim, device=env.device),  # 全1
        -torch.ones(env.num_envs, env.action_manager.total_action_dim, device=env.device),  # 全-1
        torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device),  # 全0
        torch.rand(env.num_envs, env.action_manager.total_action_dim, device=env.device) * 2.0 - 1.0,  # 随机[-1,1]
    ]
    
    test_names = ["全1动作", "全-1动作", "全0动作", "随机动作"]
    
    print(f"\n开始测试动作缩放...")
    
    for i, (actions, name) in enumerate(zip(test_actions, test_names)):
        print(f"\n--- 测试 {i+1}: {name} ---")
        print(f"输入动作范围: [{torch.min(actions):.4f}, {torch.max(actions):.4f}]")
        
        # 手动计算预期的缩放结果
        expected_scaled = unscale_transform(
            actions,
            joint_limits_lower,
            joint_limits_upper
        )
        
        # 处理动作
        env.action_manager.process_action(actions)
        
        # 获取处理后的动作
        processed_actions = action_term.processed_actions
        
        print(f"处理后动作范围: [{torch.min(processed_actions):.4f}, {torch.max(processed_actions):.4f}]")
        print(f"预期缩放范围: [{torch.min(expected_scaled):.4f}, {torch.max(expected_scaled):.4f}]")
        
        # 检查是否匹配
        diff = torch.abs(processed_actions - expected_scaled)
        max_diff = torch.max(diff)
        print(f"差异: 最大={max_diff:.6f}")
        
        if max_diff < 1e-5:
            print("✅ 动作缩放正确")
        else:
            print("❌ 动作缩放异常")
            
        # 执行一步来看实际效果
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # 获取当前关节位置
        current_joint_pos = robot.data.joint_pos
        joint_targets = robot.data.joint_pos_target
        
        print(f"关节目标范围: [{torch.min(joint_targets):.4f}, {torch.max(joint_targets):.4f}]")
        print(f"当前关节位置范围: [{torch.min(current_joint_pos):.4f}, {torch.max(current_joint_pos):.4f}]")
        
        # 检查关节目标是否在限制范围内
        within_limits = torch.all(
            (joint_targets >= joint_limits_lower - 0.01) & 
            (joint_targets <= joint_limits_upper + 0.01)
        )
        print(f"关节目标在限制内: {within_limits}")
        
        if i == 0:  # 只在第一次测试时显示详细信息
            print(f"\n详细信息 (第一个环境前4个关节):")
            print(f"  输入动作: {actions[0, :4].cpu().numpy()}")
            print(f"  处理后动作: {processed_actions[0, :4].cpu().numpy()}")
            print(f"  预期缩放: {expected_scaled[0, :4].cpu().numpy()}")
            print(f"  关节目标: {joint_targets[0, :4].cpu().numpy()}")
            print(f"  当前位置: {current_joint_pos[0, :4].cpu().numpy()}")
            print(f"  关节下限: {joint_limits_lower[0, :4].cpu().numpy()}")
            print(f"  关节上限: {joint_limits_upper[0, :4].cpu().numpy()}")
    
    print(f"\n" + "=" * 80)
    print("调试完成!")
    print("=" * 80)
    
    # 关闭环境
    env.close()


def main():
    """主函数"""
    try:
        debug_action_scaling()
    except Exception as e:
        print(f"调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭仿真
        simulation_app.close()


if __name__ == "__main__":
    main()
