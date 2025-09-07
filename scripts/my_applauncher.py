#!/usr/bin/env python3

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="通过设定初始关节位置来测试LeapHand机器人的关节索引名称及关节上下限")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import time
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from leaphand.robots.leap import LEAP_HAND_CFG
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass

@configclass
class LeapHandTestSceneCfg(InteractiveSceneCfg):
    """LeapHand测试场景配置"""
    
    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # LeapHand机器人
    leap: ArticulationCfg = LEAP_HAND_CFG.replace(
        prim_path="/World/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),  # pos,rot场景初始化时便被应用
            rot=(0.5, 0.5, -0.5, 0.5),
            joint_pos={ 
                "a_0": 0.0, "a_1": 0.0, "a_2": 0.0, "a_3": 0.0,
                "a_4": 0.0, "a_5": 0.0, "a_6": 0.0, "a_7": 0.0,
                "a_8": 0.0, "a_9": 0.0, "a_10": 0.0, "a_11": 0.0,
                "a_12": 0.0, "a_13": 0.0, "a_14": 0.0, "a_15": 0.0
            },
            joint_vel={"a_.*": 0.0},
        )
    )

def run_simple_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop 仅画面渲染."""
    # Extract scene entities
    # note: we only do this here for readability.
    # robot = scene["leap"]
    # data = robot.data
    
    # Define simulation stepping
    while simulation_app.is_running():
        # perform step
        sim.step()

def run_joint_mapping_demo(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """演示关节序列与索引的对应关系，逐一激活每个关节"""
    robot = scene["leap"]
    data = robot.data
    
    # 获取关节信息
    joint_names = robot.joint_names
    num_joints = len(joint_names)
    
    print(f"\n=== LeapHand关节映射演示 ===")
    print(f"总关节数: {num_joints}")
    print("关节名称和索引对应关系:")
    for i, joint_name in enumerate(joint_names):
        print(f"  索引 {i:2d}: {joint_name}")
    print("\n开始演示 - 每个关节将依次激活...")
    
    # 获取关节限制
    joint_limits = robot.data.joint_pos_limits[0]  # (num_joints, 2) [lower, upper]
    
    # 基础位置（所有关节为0）
    base_positions = torch.zeros((1, num_joints), device=robot.device)
    base_velocities = torch.zeros((1, num_joints), device=robot.device)
    
    current_joint_idx = 0
    step_counter = 0
    hold_time = 180  # 每个关节保持激活状态的帧数（约3秒）
    
    while simulation_app.is_running():
        # 重置到基础位置
        positions = base_positions.clone()
        
        # 激活当前关节
        if current_joint_idx < num_joints:
            joint_name = joint_names[current_joint_idx]
            lower_limit = joint_limits[current_joint_idx, 0].item()
            upper_limit = joint_limits[current_joint_idx, 1].item()
            
            # 设置关节到中间位置或安全位置
            if upper_limit > 0.1:  # 如果上限大于0.1，设置为上限的60%
                target_pos = upper_limit * 0.6
            elif lower_limit < -0.1:  # 如果下限小于-0.1，设置为下限的60%
                target_pos = lower_limit * 0.6
            else:  # 否则设置为可用范围的中点
                target_pos = (upper_limit + lower_limit) / 2
            
            positions[0, current_joint_idx] = target_pos
            
            # 每隔30帧打印一次当前状态
            if step_counter % 30 == 0:
                print(f"正在演示关节 {current_joint_idx:2d}: {joint_name:6s} -> 位置: {target_pos:6.3f} (范围: [{lower_limit:6.3f}, {upper_limit:6.3f}])")
        
        # 应用关节位置 （验证position中的序列位置与实际关节索引的映射关系）
        robot.write_joint_state_to_sim(positions, base_velocities)
        
        # 步进仿真
        sim.step()
        step_counter += 1
        
        # 检查是否切换到下一个关节
        if step_counter >= hold_time:
            current_joint_idx += 1
            step_counter = 0
            
            if current_joint_idx >= num_joints:
                print("\n所有关节演示完成，重新开始...")
                current_joint_idx = 0
                # 短暂停留在基础位置
                robot.write_joint_state_to_sim(base_positions, base_velocities)
                for _ in range(60):  # 停留1秒
                    sim.step()

def run_goal_reaching_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop 设定关节角度."""
    pass

def main():
    """主函数"""
    
    # Create the simulation context
    sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, render_interval=1))
    
    # Create the scene
    scene_cfg = LeapHandTestSceneCfg(num_envs=1, env_spacing=1.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    # 获取机器人资产
    robot = scene["leap"]
    data = robot.data
    
    # 获取关节名称和索引
    joint_names = robot.joint_names # 这个也可以用data.joint_names
    robot.find_joints("a_.*", preserve_order=True)
    # 获取关节上下限
    joint_limits = data.default_joint_limits[0,:] # 因为default_joint_limits是(num_envs, num_joints, 2)的形状
    
    # 打印关节名称和索引
    print("\n=== 关节信息总览 ===")
    print("关节名称和索引:")
    for i, joint_name in enumerate(joint_names):
        print(f" 关节索引: {i}: 关节名称: {joint_name}")
    
    # 打印关节上下限
    print("关节上下限:")
    for joint_name, joint_limit in zip(joint_names, joint_limits):
        print(f"关节名称: {joint_name}, 关节上下限: {joint_limit}")
    
    # 启动关节映射演示
    # run_joint_mapping_demo(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close() # 这个不加就会直接关闭