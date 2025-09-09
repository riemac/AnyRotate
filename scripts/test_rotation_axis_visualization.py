#!/usr/bin/env python3
"""测试旋转轴可视化功能的脚本

用于验证旋转轴箭头是否正确显示，以及是否能够跟随课程学习动态更新。
"""

import argparse
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source'))

# 导入Isaac Lab
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="测试旋转轴可视化功能")
parser.add_argument("--num_envs", type=int, default=4, help="环境数量")
parser.add_argument("--headless", action="store_true", help="无头模式运行")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余部分在应用启动后执行"""

import torch
import time

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext

# 导入环境配置
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import (
    LeaphandContinuousRotEnvCfg
)

def test_rotation_axis_visualization():
    """测试旋转轴可视化功能"""
    
    print("=" * 60)
    print("LeapHand连续旋转环境 - 旋转轴可视化测试")
    print("=" * 60)
    
    # 创建环境配置
    env_cfg = LeaphandContinuousRotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 确保启用旋转轴可视化
    env_cfg.commands.rotation_axis.debug_vis = True
    
    print(f"环境数量: {env_cfg.scene.num_envs}")
    print(f"旋转轴可视化: {env_cfg.commands.rotation_axis.debug_vis}")
    print(f"旋转轴模式: {env_cfg.commands.rotation_axis.rotation_axis_mode}")
    
    # 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("\n✅ 环境创建成功！")
    print("📍 可视化说明：")
    print("   - 红色箭头表示当前旋转轴方向")
    print("   - 箭头位于物体上方")
    print("   - 箭头方向遵循右手螺旋定则")
    print("   - 拇指指向箭头方向，其余手指弯曲方向为正旋转方向")
    
    # 测试不同的旋转轴模式
    test_modes = ["z_axis", "x_axis", "y_axis", "random"]
    
    try:
        # 重置环境
        env.reset()
        
        for mode in test_modes:
            print(f"\n🔄 测试旋转轴模式: {mode}")
            
            # 更新旋转轴模式
            env.command_manager.get_term("rotation_axis").cfg.rotation_axis_mode = mode
            
            # 重新采样命令以应用新模式
            env_ids = torch.arange(env.num_envs, device=env.device)
            env.command_manager.get_term("rotation_axis")._resample_command(env_ids)
            
            # 运行几步以观察可视化
            for step in range(50):
                # 执行随机动作
                actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
                actions = torch.clamp(actions, -1.0, 1.0)
                
                # 执行步骤
                env.step(actions)
                
                # 每10步打印一次旋转轴信息
                if step % 10 == 0:
                    rotation_axes = env.command_manager.get_term("rotation_axis").command
                    print(f"   步骤 {step:2d}: 旋转轴 = {rotation_axes[0].cpu().numpy()}")
            
            # 等待用户观察
            if not args_cli.headless:
                print(f"   ⏸️  请在Isaac Sim中观察 {mode} 模式的旋转轴可视化")
                print("      按Enter键继续下一个模式...")
                input()
    
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
    finally:
        # 关闭环境
        env.close()
        print("\n✅ 测试完成！")

def main():
    """主函数"""
    try:
        test_rotation_axis_visualization()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return 1
    return 0

if __name__ == "__main__":
    # 运行测试
    exit_code = main()
    # 关闭仿真应用
    simulation_app.close()
    sys.exit(exit_code)
