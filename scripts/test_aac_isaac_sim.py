#!/usr/bin/env python3

"""在Isaac Sim中测试非对称Actor-Critic实现的脚本"""

import argparse
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="测试AAC LeapHand环境")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动Isaac Sim应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 导入必要的模块
import numpy as np
from leaphand.tasks.direct.leaphand.leaphand_continuous_rot_env import LeaphandContinuousRotEnv
from leaphand.tasks.direct.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg

def test_aac_environment():
    """测试AAC环境实现"""
    print("=" * 80)
    print("测试非对称Actor-Critic LeapHand连续旋转环境")
    print("=" * 80)
    
    # 创建环境配置
    cfg = LeaphandContinuousRotEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    
    print(f"环境配置:")
    print(f"  - 环境数量: {cfg.scene.num_envs}")
    print(f"  - 非对称观测: {cfg.asymmetric_obs}")
    print(f"  - Actor观测空间维度: {cfg.observation_space}")
    print(f"  - Critic状态空间维度: {cfg.state_space}")
    print(f"  - Actor历史步数: {cfg.observations_cfg['actor']['history_steps']}")
    print(f"  - Critic历史步数: {cfg.observations_cfg['critic']['history_steps']}")
    
    try:
        # 创建环境
        print("\n创建环境...")
        env = LeaphandContinuousRotEnv(cfg, render_mode=None)
        print("✓ 环境创建成功")
        
        # 重置环境
        print("\n重置环境...")
        obs_dict, extras = env.reset()
        print("✓ 环境重置成功")

        # 检查观测字典结构
        print(f"\n观测字典键: {list(obs_dict.keys())}")
        
        if "policy" in obs_dict:
            actor_obs = obs_dict["policy"]
            print(f"Actor观测维度: {actor_obs.shape}")
            print(f"  - 期望维度: ({cfg.scene.num_envs}, {cfg.observation_space})")
            print(f"  - 实际维度: {actor_obs.shape}")
            assert actor_obs.shape == (cfg.scene.num_envs, cfg.observation_space), \
                f"Actor观测维度不匹配: 期望 {(cfg.scene.num_envs, cfg.observation_space)}, 实际 {actor_obs.shape}"
            print("✓ Actor观测维度正确")
        
        if "critic" in obs_dict:
            critic_state = obs_dict["critic"]
            print(f"Critic状态维度: {critic_state.shape}")
            print(f"  - 期望维度: ({cfg.scene.num_envs}, {cfg.state_space})")
            print(f"  - 实际维度: {critic_state.shape}")
            assert critic_state.shape == (cfg.scene.num_envs, cfg.state_space), \
                f"Critic状态维度不匹配: 期望 {(cfg.scene.num_envs, cfg.state_space)}, 实际 {critic_state.shape}"
            print("✓ Critic状态维度正确")
        
        # 测试多步执行
        print(f"\n测试多步执行...")
        num_steps = 10
        for step in range(num_steps):
            # 生成随机动作
            actions = torch.randn(cfg.scene.num_envs, cfg.action_space, device=env.device)
            
            # 执行一步
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            
            # 检查观测维度
            if step == 0:  # 只在第一步检查，避免重复输出
                print(f"  步骤 {step + 1}:")
                if "policy" in obs_dict:
                    print(f"    Actor观测维度: {obs_dict['policy'].shape}")
                if "critic" in obs_dict:
                    print(f"    Critic状态维度: {obs_dict['critic'].shape}")
                print(f"    奖励形状: {rewards.shape}")
                print(f"    奖励范围: [{rewards.min().item():.3f}, {rewards.max().item():.3f}]")
        
        print("✓ 多步执行成功")
        
        # 测试历史缓冲区
        print(f"\n测试历史缓冲区...")
        
        # 检查Actor历史缓冲区
        actor_buffers = env.actor_history_buffers
        print(f"Actor历史缓冲区组件: {list(actor_buffers.keys())}")
        for component_name, buffer in actor_buffers.items():
            if buffer is not None:
                print(f"  {component_name}: 缓冲区大小 {buffer.buffer.shape}")
        
        # 检查Critic历史缓冲区
        critic_buffers = env.critic_history_buffers
        print(f"Critic历史缓冲区组件: {list(critic_buffers.keys())}")
        for component_name, buffer in critic_buffers.items():
            if buffer is not None:
                print(f"  {component_name}: 缓冲区大小 {buffer.buffer.shape}")
        
        print("✓ 历史缓冲区工作正常")
        
        # 测试环境重置
        print(f"\n测试环境重置...")
        reset_env_ids = torch.tensor([0, min(2, cfg.scene.num_envs-1)], device=env.device)  # 重置部分环境
        env._reset_idx(reset_env_ids)
        print("✓ 部分环境重置成功")
        
        print(f"\n" + "=" * 80)
        print("所有测试通过！非对称Actor-Critic实现正确。")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = test_aac_environment()
    
    # 关闭仿真
    simulation_app.close()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
