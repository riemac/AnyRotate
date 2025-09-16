#!/usr/bin/env python3

"""调试LeapHand机器人的关节索引名称 - 简化版本。该脚本仅用于测试，没有实际的仿真环境。"""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="调试LeapHand机器人的关节索引名称")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from leaphand.robots.leap import LEAP_HAND_CFG


def main():
    """主函数"""
    
    # 启动仿真上下文
    sim = SimulationContext()
    
    # 设置主舞台
    # stage = sim.stage
    
    # 添加地面平面
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    
    # 添加光源
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    
    # 创建机器人配置副本并设置路径
    robot_cfg = LEAP_HAND_CFG.replace(prim_path="/World/Robot")
    
    # 从USD文件中产生机器人
    robot_cfg.spawn.func(robot_cfg.prim_path, robot_cfg.spawn)
    
    # 创建机器人实例
    robot = Articulation(cfg=robot_cfg)
    
    # 初始化仿真
    sim.reset()
    
    print("=" * 60)
    print("LeapHand 关节索引名称调试信息")
    print("=" * 60)
    
    # 打印基本信息
    print(f"机器人资产路径: {robot_cfg.spawn.usd_path}")
    print(f"机器人路径: {robot_cfg.prim_path}")
    print(f"总关节数量: {robot.num_joints}")
    print(f"执行器关节数量: {len(robot.joint_names) if hasattr(robot, 'joint_names') else 'N/A'}")
    
    print("\n" + "=" * 60)
    print("关节名称列表")
    print("=" * 60)
    
    # 打印所有关节名称和索引
    joint_names = robot.joint_names
    for i, joint_name in enumerate(joint_names):
        print(f"索引 {i:2d}: {joint_name}")
    
    print("\n" + "=" * 60)
    print("体(Body)名称列表")
    print("=" * 60)
    
    # 打印所有体名称和索引
    body_names = robot.body_names
    for i, body_name in enumerate(body_names):
        print(f"索引 {i:2d}: {body_name}")
    
    print("\n" + "=" * 60)
    print("关节位置限制")
    print("=" * 60)
    
    # 打印关节位置限制
    try:
        joint_pos_limits = robot.root_physx_view.get_dof_limits()
        if len(joint_pos_limits) >= 2:
            for i, joint_name in enumerate(joint_names):
                lower_limit = joint_pos_limits[0][0][i].item()
                upper_limit = joint_pos_limits[1][0][i].item()
                print(f"{joint_name:20s}: [{lower_limit:8.4f}, {upper_limit:8.4f}]")
        else:
            print("无法获取关节位置限制")
    except Exception as e:
        print(f"获取关节位置限制时出错: {e}")
    
    print("\n" + "=" * 60)
    print("当前关节位置")
    print("=" * 60)
    
    # 打印当前关节位置
    joint_pos = robot.data.joint_pos
    for i, joint_name in enumerate(joint_names):
        pos = joint_pos[0][i].item()
        print(f"{joint_name:20s}: {pos:8.4f}")
    
    print("\n" + "=" * 60)
    print("关节名称分组分析")
    print("=" * 60)
    
    # 分析关节名称的模式
    finger_joints = {}
    other_joints = []
    
    for i, joint_name in enumerate(joint_names):
        # 尝试识别手指关节模式
        if 'thumb' in joint_name.lower():
            finger = 'thumb'
        elif 'index' in joint_name.lower():
            finger = 'index'
        elif 'middle' in joint_name.lower():
            finger = 'middle'
        elif 'ring' in joint_name.lower():
            finger = 'ring'
        elif 'pinky' in joint_name.lower() or 'little' in joint_name.lower():
            finger = 'pinky'
        else:
            # 尝试从关节名称中提取手指信息，或按数字索引分类
            if joint_name.startswith('a_'):
                # LeapHand通常使用a_0, a_1等命名方式
                joint_num = int(joint_name.split('_')[1]) if '_' in joint_name else -1
                if 0 <= joint_num <= 3:
                    finger = 'index'
                elif 4 <= joint_num <= 7:
                    finger = 'middle'
                elif 8 <= joint_num <= 11:
                    finger = 'ring'
                elif 12 <= joint_num <= 15:
                    finger = 'thumb'
                else:
                    finger = 'unknown'
            else:
                finger = 'unknown'
        
        if finger != 'unknown':
            if finger not in finger_joints:
                finger_joints[finger] = []
            finger_joints[finger].append((i, joint_name))
        else:
            other_joints.append((i, joint_name))
    
    # 按手指分组显示
    for finger, joints in finger_joints.items():
        print(f"{finger.capitalize()} 手指关节:")
        for idx, name in joints:
            print(f"  索引 {idx:2d}: {name}")
        print()
    
    if other_joints:
        print("其他关节:")
        for idx, name in other_joints:
            print(f"  索引 {idx:2d}: {name}")
    
    print("\n" + "=" * 60)
    print("关节名称匹配测试")
    print("=" * 60)
    
    # 测试常用的关节名称模式匹配
    test_patterns = [
        "a_.*",  # 配置文件中使用的模式
        "a_[0-3]",  # 食指（假设）
        "a_[4-7]",  # 中指（假设）
        "a_[8-9]|a_1[01]", # 无名指（假设）
        "a_1[2-5]", # 拇指（假设）
    ]
    
    import re
    for pattern in test_patterns:
        regex = re.compile(pattern)
        matching_joints = [name for name in joint_names if regex.match(name)]
        print(f"模式 '{pattern}' 匹配的关节:")
        if matching_joints:
            for joint in matching_joints:
                print(f"  {joint}")
        else:
            print("  无匹配")
        print()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
