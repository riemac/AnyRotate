#!/usr/bin/env python3

"""
LeapHand连续旋转任务课程学习演示脚本

该脚本展示了如何使用不同的课程学习配置：
1. 无课程学习
2. 仅奖励权重课程学习
3. 仅自适应域随机化课程学习
4. 仅旋转轴复杂度课程学习
5. 完整课程学习
6. 自定义课程学习

使用方法:
    python scripts/curriculum_demo.py --env_cfg <config_name>

可用的配置名称:
    - no_curriculum: 无课程学习
    - reward_only: 仅奖励权重课程学习
    - adr_only: 仅自适应域随机化课程学习
    - axis_only: 仅旋转轴复杂度课程学习
    - simple_axis: 简化旋转轴课程学习
    - full_curriculum: 完整课程学习（默认）

作者: AI Assistant
日期: 2025-01-05
"""

import argparse
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="LeapHand连续旋转任务课程学习演示")
parser.add_argument("--num_envs", type=int, default=64, help="环境数量")
parser.add_argument("--env_cfg", type=str, default="full_curriculum",
                   choices=["no_curriculum", "reward_only", "adr_only", "axis_only",
                           "simple_axis", "full_curriculum"],
                   help="课程学习配置类型")
parser.add_argument("--max_steps", type=int, default=10000, help="最大仿真步数")

# 启动应用
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 导入必要的模块
from isaaclab.envs import ManagerBasedRLEnv

# 导入环境配置
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import (
    LeaphandContinuousRotEnvCfg,
    LeaphandContinuousRotFullCurriculumEnvCfg,
    LeaphandContinuousRotRewardOnlyEnvCfg,
    LeaphandContinuousRotADROnlyEnvCfg,
    LeaphandContinuousRotAxisOnlyEnvCfg,
    LeaphandContinuousRotSimpleAxisEnvCfg
)


def main():
    """主函数"""
    
    # 配置映射
    config_map = {
        "no_curriculum": LeaphandContinuousRotEnvCfg,  # 默认无课程学习
        "reward_only": LeaphandContinuousRotRewardOnlyEnvCfg,
        "adr_only": LeaphandContinuousRotADROnlyEnvCfg,
        "axis_only": LeaphandContinuousRotAxisOnlyEnvCfg,
        "simple_axis": LeaphandContinuousRotSimpleAxisEnvCfg,
        "full_curriculum": LeaphandContinuousRotFullCurriculumEnvCfg
    }
    
    # 获取配置类
    env_cfg_class = config_map[args_cli.env_cfg]
    env_cfg = env_cfg_class()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    print(f"\n{'='*60}")
    print(f"LeapHand连续旋转任务课程学习演示")
    print(f"{'='*60}")
    print(f"配置类型: {args_cli.env_cfg}")
    print(f"环境数量: {args_cli.num_envs}")
    print(f"最大步数: {args_cli.max_steps}")
    print(f"{'='*60}\n")
    
    # 打印课程学习配置信息
    print("课程学习配置:")
    if hasattr(env_cfg, 'curriculum') and env_cfg.curriculum is not None:
        curriculum_terms = []
        for attr_name in dir(env_cfg.curriculum):
            if not attr_name.startswith('_'):
                attr_value = getattr(env_cfg.curriculum, attr_name)
                if hasattr(attr_value, 'func'):  # 是课程学习项
                    curriculum_terms.append(attr_name)
        
        if curriculum_terms:
            print(f"  启用的课程学习项: {', '.join(curriculum_terms)}")
        else:
            print("  无课程学习项")
    else:
        print("  无课程学习配置")
    
    print(f"\n{'='*60}\n")
    
    # 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 打印环境信息
    print("环境信息:")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  奖励项数量: {len(env.reward_manager.active_terms)}")
    print(f"  终止项数量: {len(env.termination_manager.active_terms)}")
    print(f"  事件项数量: {len(env.event_manager.active_terms)}")
    
    if hasattr(env, 'curriculum_manager') and env.curriculum_manager is not None:
        print(f"  课程学习项数量: {len(env.curriculum_manager.active_terms)}")
        print(f"  课程学习项: {env.curriculum_manager.active_terms}")
    else:
        print("  课程学习项数量: 0")
    
    print(f"\n{'='*60}\n")
    
    # 重置环境
    obs, _ = env.reset()
    print(f"初始观测形状: {obs['policy'].shape}")
    
    # 运行仿真
    print("开始仿真...")
    step_count = 0
    
    try:
        while step_count < args_cli.max_steps:
            # 随机动作
            actions = torch.randn(env.num_envs, env.action_space.shape[0], device=env.device)
            
            # 执行动作
            obs, rewards, terminated, truncated, _ = env.step(actions)
            
            step_count += 1
            
            # 每1000步打印一次信息
            if step_count % 1000 == 0:
                print(f"步数: {step_count}")
                print(f"  平均奖励: {rewards.mean().item():.4f}")
                print(f"  终止环境数: {terminated.sum().item()}")
                print(f"  截断环境数: {truncated.sum().item()}")
                
                # 打印课程学习状态
                if hasattr(env, 'curriculum_manager') and env.curriculum_manager is not None:
                    curriculum_state = env.curriculum_manager.get_state()
                    if curriculum_state:
                        print(f"  课程学习状态: {curriculum_state}")
                
                print()
            
            # 重置已终止的环境
            if terminated.any() or truncated.any():
                reset_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    env.reset(reset_ids)
    
    except KeyboardInterrupt:
        print("\n仿真被用户中断")
    
    finally:
        # 关闭环境
        env.close()
        print(f"\n仿真完成，总步数: {step_count}")
        print("环境已关闭")


if __name__ == "__main__":
    main()
    simulation_app.close()
