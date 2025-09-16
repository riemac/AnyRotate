#!/usr/bin/env python3

"""测试旋转轴箭头可视化效果的脚本

该脚本用于验证修复后的箭头可视化是否正常工作：
1. 每个环境都有一个箭头
2. 箭头位置正确（考虑env_origins偏移）
3. 随机模式下不同环境有不同的旋转轴
4. 箭头外观改进（更大、更鲜艳）
"""

import argparse
import torch

from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="测试旋转轴箭头可视化")
parser.add_argument("--num_envs", type=int, default=4, help="环境数量")
parser.add_argument("--rotation_axis_mode", type=str, default="random",
                   choices=["z_axis", "x_axis", "y_axis", "random"],
                   help="旋转轴模式")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余导入"""
import time
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg

def main():
    """主函数"""
    
    # 创建环境配置
    env_cfg = LeaphandContinuousRotEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # 设置旋转轴模式
    env_cfg.commands.rotation_axis.rotation_axis_mode = args_cli.rotation_axis_mode
    env_cfg.commands.rotation_axis.debug_vis = True  # 确保启用可视化
    
    # 创建环境
    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print(f"创建了 {env.num_envs} 个环境")
    print(f"旋转轴模式: {args_cli.rotation_axis_mode}")
    print("箭头可视化已启用")
    
    # 重置环境
    env.reset()
    
    # 打印每个环境的旋转轴信息
    rotation_axes = env.command_manager.get_command("rotation_axis")
    print("\n各环境的旋转轴:")
    for i in range(env.num_envs):
        axis = rotation_axes[i].cpu().numpy()
        print(f"环境 {i}: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
    
    # 验证随机模式下轴是否不同
    if args_cli.rotation_axis_mode == "random":
        axes_tensor = rotation_axes
        # 计算轴之间的差异
        differences = []
        for i in range(env.num_envs):
            for j in range(i+1, env.num_envs):
                diff = torch.norm(axes_tensor[i] - axes_tensor[j]).item()
                differences.append(diff)
        
        avg_diff = sum(differences) / len(differences) if differences else 0
        print(f"\n随机轴验证:")
        print(f"平均轴差异: {avg_diff:.3f}")
        if avg_diff > 0.1:
            print("✓ 随机轴分配正确 - 不同环境有不同的旋转轴")
        else:
            print("✗ 随机轴分配可能有问题 - 轴过于相似")
    
    print("\n开始仿真循环...")
    print("请在Isaac Sim中观察箭头可视化效果:")
    print("- 每个环境应该有一个红色箭头")
    print("- 箭头应该位于物体上方")
    print("- 箭头方向应该指示旋转轴方向")
    print("按 Ctrl+C 退出")
    
    # 仿真循环
    try:
        step_count = 0
        while simulation_app.is_running():
            # 生成随机动作
            actions = torch.randn(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            actions = torch.clamp(actions, -1.0, 1.0)
            
            # 执行步骤
            env.step(actions)
            
            # 每100步重置一次环境以测试箭头更新
            step_count += 1
            if step_count % 100 == 0:
                print(f"步骤 {step_count}: 重置部分环境以测试箭头更新")
                # 重置前两个环境
                env.reset(env_ids=torch.tensor([0, 1], device=env.device))
                
                # 打印更新后的旋转轴
                rotation_axes = env.command_manager.get_command("rotation_axis")
                print("重置后的旋转轴:")
                for i in [0, 1]:
                    axis = rotation_axes[i].cpu().numpy()
                    print(f"环境 {i}: [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
            
            # 控制仿真速度
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\n收到中断信号，正在退出...")
    
    # 清理
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
