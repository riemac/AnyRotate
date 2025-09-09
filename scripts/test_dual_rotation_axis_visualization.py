#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
测试LeapHand环境中的双旋转轴可视化功能：
- 红色箭头：目标旋转轴（来自Command管理器）
- 蓝色箭头：实际旋转轴（来自Reward计算）

使用方法:
python scripts/test_dual_rotation_axis_visualization.py --num_envs 4 --rotation_axis_mode random
"""

import argparse
import sys
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="测试LeapHand双旋转轴可视化")
parser.add_argument("--num_envs", type=int, default=4, help="环境数量")
parser.add_argument("--rotation_axis_mode", type=str, default="random", 
                   choices=["z_axis", "x_axis", "y_axis", "random"],
                   help="旋转轴模式")
parser.add_argument("--headless", action="store_true", help="无头模式运行")

# 解析参数并启动应用
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim import SimulationContext

# 导入环境配置
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import (
    LeaphandContinuousRotEnvCfg
)

def test_dual_rotation_axis_visualization():
    """测试双旋转轴可视化功能"""
    
    print("=" * 80)
    print("LeapHand连续旋转环境 - 双旋转轴可视化测试")
    print("=" * 80)
    
    # 创建环境配置
    env_cfg = LeaphandContinuousRotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 设置旋转轴模式
    env_cfg.commands.rotation_axis.rotation_axis_mode = args_cli.rotation_axis_mode
    
    # 确保启用目标旋转轴可视化
    env_cfg.commands.rotation_axis.debug_vis = True
    
    print(f"环境数量: {env_cfg.scene.num_envs}")
    print(f"旋转轴模式: {args_cli.rotation_axis_mode}")
    print(f"目标旋转轴可视化: {env_cfg.commands.rotation_axis.debug_vis}")
    print(f"实际旋转轴可视化: 已在奖励函数中启用")
    
    # 创建环境
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("\n✅ 环境创建成功！")
    print("📍 可视化说明：")
    print("   🔴 红色箭头：目标旋转轴（Command管理器指定的目标方向）")
    print("   🔵 蓝色箭头：实际旋转轴（物体实际旋转的瞬时轴）")
    print("   📏 箭头位置：物体上方（蓝色箭头稍高于红色箭头）")
    print("   🔄 箭头方向：遵循右手螺旋定则")
    print("   ⚠️  注意：蓝色箭头仅在物体有有效旋转时显示")
    
    try:
        # 重置环境
        env.reset()
        
        print(f"\n🔄 开始测试旋转轴模式: {args_cli.rotation_axis_mode}")
        
        # 打印初始旋转轴信息
        rotation_axes = env.command_manager.get_command("rotation_axis")
        print("\n📊 各环境的目标旋转轴:")
        for i in range(min(env.num_envs, 8)):  # 最多显示8个环境的信息
            axis = rotation_axes[i].cpu().numpy()
            print(f"   环境 {i}: [{axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}]")
        
        if env.num_envs > 8:
            print(f"   ... (还有 {env.num_envs - 8} 个环境)")
        
        # 运行仿真步骤
        print(f"\n🚀 开始运行仿真...")
        print("   💡 提示：观察Isaac Sim中的双色箭头可视化")
        print("   💡 红色箭头应该相对稳定（目标轴）")
        print("   💡 蓝色箭头会根据物体实际旋转动态变化")
        
        step_count = 0
        max_steps = 1000
        
        while step_count < max_steps:
            # 执行随机动作
            actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            actions = torch.clamp(actions, -1.0, 1.0)
            
            # 执行步骤
            env.step(actions)
            step_count += 1
            
            # 每100步打印一次信息
            if step_count % 100 == 0:
                print(f"   步骤 {step_count:4d}/{max_steps}: 可视化正在更新...")
                
                # 打印第一个环境的旋转轴信息
                target_axis = rotation_axes[0].cpu().numpy()
                print(f"     环境0目标轴: [{target_axis[0]:+.3f}, {target_axis[1]:+.3f}, {target_axis[2]:+.3f}]")
            
            # 在无头模式下运行更少的步骤
            if args_cli.headless and step_count >= 200:
                break
        
        print(f"\n✅ 测试完成！共运行 {step_count} 步")
        
        if not args_cli.headless:
            print("\n⏸️  测试已完成，但环境仍在运行")
            print("   💡 您可以继续在Isaac Sim中观察可视化效果")
            print("   💡 按Ctrl+C退出程序")
            
            # 保持环境运行，让用户观察
            try:
                while True:
                    actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
                    actions = torch.clamp(actions, -0.5, 0.5)  # 使用较小的动作幅度
                    env.step(actions)
            except KeyboardInterrupt:
                print("\n⏹️  用户中断，正在退出...")
    
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭环境
        env.close()
        print("\n✅ 环境已关闭！")

def main():
    """主函数"""
    try:
        test_dual_rotation_axis_visualization()
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
