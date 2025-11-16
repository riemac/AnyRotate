#!/usr/bin/env python3

"""
启动 Isaac Sim 中的 LeapHand 场景，并提供实时位姿监控UI面板。

使用方法:
    ./isaaclab.sh -p source/leaphand/leaphand/tasks/manager_based/launch_with_leaphand.py
"""

import argparse
from isaaclab.app import AppLauncher

# 添加命令行参数
parser = argparse.ArgumentParser(description="启动LeapHand场景并显示实时位姿监控面板")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.ui as ui
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from leaphand.robots.leap import LEAP_HAND_CFG


class LeapHandPoseMonitor:
    """LeapHand实时位姿监控面板"""
    
    def __init__(self, robot):
        """初始化监控面板
        
        Args:
            robot: LeapHand机器人实例
        """
        self.robot = robot
        self.body_names = robot.data.body_names
        
        # 创建UI窗口
        self._window = ui.Window(
            "LeapHand Body Pose Monitor", 
            width=800, 
            height=600,
            flags=ui.WINDOW_FLAGS_NO_COLLAPSE
        )
        
        # 存储UI标签引用
        self.pose_labels = {}
        
        with self._window.frame:
            with ui.ScrollingFrame(
                height=ui.Fraction(1),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                with ui.VStack(spacing=10, height=0):
                    # 标题
                    ui.Label("LeapHand Bodies - World Frame Poses", 
                            height=30, 
                            style={"font_size": 18, "color": 0xFF00FFFF})
                    
                    ui.Separator(height=2)
                    
                    # 为每个body创建显示行
                    for body_name in self.body_names:
                        self._create_body_row(body_name)
    
    def _create_body_row(self, body_name: str):
        """为单个body创建位姿显示行
        
        Args:
            body_name: Body名称
        """
        with ui.CollapsableFrame(
            title=body_name,
            height=0,
            collapsed=True,
            style={"color": 0xFFCCCCCC}
        ):
            with ui.VStack(spacing=5, height=0):
                # 位置显示
                with ui.HStack(spacing=10, height=20):
                    ui.Label("Position (m):", width=120, style={"color": 0xFF88FF88})
                    pos_label = ui.Label("", style={"color": 0xFFFFFFFF})
                
                # 姿态显示 (四元数 wxyz)
                with ui.HStack(spacing=10, height=20):
                    ui.Label("Orientation (quat):", width=120, style={"color": 0xFFFF8888})
                    quat_label = ui.Label("", style={"color": 0xFFFFFFFF})
                
                # 存储标签引用
                self.pose_labels[body_name] = {
                    "pos": pos_label,
                    "quat": quat_label
                }
    
    def update(self):
        """更新所有body的位姿显示（每帧调用）"""
        # 获取所有body的世界坐标系位姿
        body_poses_w = self.robot.data.body_pose_w  # shape: (num_envs, num_bodies, 7)
        
        # 只显示第一个环境的数据
        for idx, body_name in enumerate(self.body_names):
            if body_name in self.pose_labels:
                # 提取位置和姿态 (第0个环境，第idx个body)
                pos = body_poses_w[0, idx, :3].cpu().numpy()
                quat = body_poses_w[0, idx, 3:7].cpu().numpy()  # (w, x, y, z)
                
                # 更新UI标签
                pos_str = f"x: {pos[0]:7.4f}  y: {pos[1]:7.4f}  z: {pos[2]:7.4f}"
                quat_str = f"w: {quat[0]:6.3f}  x: {quat[1]:6.3f}  y: {quat[2]:6.3f}  z: {quat[3]:6.3f}"
                
                self.pose_labels[body_name]["pos"].text = pos_str
                self.pose_labels[body_name]["quat"].text = quat_str


class LeapHandControlPanel:
    """LeapHand关节控制面板
    
    提供16个关节的滑动条控制，按手指分组显示。
    """
    
    def __init__(self, robot):
        """初始化控制面板
        
        Args:
            robot: LeapHand机器人实例
        """
        self.robot = robot
        self.joint_names = robot.joint_names
        self.num_joints = len(self.joint_names)
        
        # 获取关节限制 (num_joints, 2) - [lower, upper]
        self.joint_limits = robot.data.joint_pos_limits[0].cpu()
        
        # 存储当前目标位置
        self.joint_targets = torch.zeros(1, self.num_joints, device=robot.device)
        
        # 创建UI窗口
        self._window = ui.Window(
            "LeapHand Joint Control", 
            width=400, 
            height=700,
            flags=ui.WINDOW_FLAGS_NO_COLLAPSE,
            dock_preference=ui.DockPreference.LEFT_BOTTOM
        )
        
        # 存储滑动条引用
        self.joint_sliders = {}
        
        with self._window.frame:
            with ui.ScrollingFrame(
                height=ui.Fraction(1),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):
                with ui.VStack(spacing=10, height=0):
                    # 标题
                    ui.Label("Joint Position Control", 
                            height=30, 
                            style={"font_size": 18, "color": 0xFFFFAA00})
                    
                    ui.Separator(height=2)
                    
                    # 按手指分组创建控制面板
                    self._create_joint_groups()
                    
                    # 底部控制按钮
                    with ui.HStack(spacing=10, height=40):
                        ui.Button("Reset All", 
                                 width=0, 
                                 height=35,
                                 clicked_fn=self._reset_all_joints,
                                 style={"background_color": 0xFF4444FF})
                        ui.Button("Open Hand", 
                                 width=0, 
                                 height=35,
                                 clicked_fn=self._open_hand,
                                 style={"background_color": 0xFF44FF44})
    
    def _create_joint_groups(self):
        """创建按手指分组的关节控制"""
        # 定义手指分组 (基于LeapHand的关节命名规则)
        finger_groups = {
            "Thumb": ["a_0", "a_1", "a_2", "a_3"],
            "Index": ["a_4", "a_5", "a_6", "a_7"],
            "Middle": ["a_8", "a_9", "a_10", "a_11"],
            "Ring": ["a_12", "a_13", "a_14", "a_15"],
        }
        
        for group_name, joint_names in finger_groups.items():
            with ui.CollapsableFrame(
                title=group_name,
                height=0,
                collapsed=False,
                style={"color": 0xFFAAFFAA}
            ):
                with ui.VStack(spacing=5, height=0):
                    for joint_name in joint_names:
                        if joint_name in self.joint_names:
                            joint_idx = self.joint_names.index(joint_name)
                            self._create_joint_slider(joint_name, joint_idx)
    
    def _create_joint_slider(self, joint_name: str, joint_idx: int):
        """为单个关节创建滑动条
        
        Args:
            joint_name: 关节名称
            joint_idx: 关节索引
        """
        lower = self.joint_limits[joint_idx, 0].item()
        upper = self.joint_limits[joint_idx, 1].item()
        
        with ui.HStack(spacing=5, height=25):
            # 关节名称标签
            ui.Label(f"{joint_name}:", width=60, style={"color": 0xFFCCCCCC})
            
            # 滑动条
            slider = ui.FloatSlider(
                min=lower,
                max=upper,
                width=ui.Fraction(0.6),
                height=20
            )
            
            # 数值显示标签
            value_label = ui.Label(
                f"{0.0:6.3f}", 
                width=60, 
                alignment=ui.Alignment.CENTER,
                style={"color": 0xFFFFFFFF}
            )
            
            # 绑定回调函数
            def on_value_changed(model, idx=joint_idx, label=value_label):
                value = model.as_float
                # 更新标签
                label.text = f"{value:6.3f}"
                # 更新目标位置
                self.joint_targets[0, idx] = value
                # 立即应用到机器人
                self.robot.set_joint_position_target(self.joint_targets)
            
            slider.model.add_value_changed_fn(on_value_changed)
            
            # 存储引用
            self.joint_sliders[joint_name] = {
                "slider": slider,
                "label": value_label,
                "index": joint_idx
            }
    
    def _reset_all_joints(self):
        """重置所有关节到零位"""
        self.joint_targets.zero_()
        
        # 更新UI滑动条
        for joint_data in self.joint_sliders.values():
            joint_data["slider"].model.set_value(0.0)
            joint_data["label"].text = " 0.000"
        
        # 应用到机器人
        self.robot.set_joint_position_target(self.joint_targets)
        print("[INFO] 所有关节已重置到零位")
    
    def _open_hand(self):
        """张开手掌（设置所有关节到零位，LeapHand零位即为张开状态）"""
        # LeapHand的零位就是张开状态
        self._reset_all_joints()


@configclass
class LeapHandSceneCfg(InteractiveSceneCfg):
    """LeapHand场景配置"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # 光源
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # LeapHand机器人
    robot = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """运行仿真循环
    
    Args:
        sim: 仿真上下文
        scene: 交互式场景
    """
    # 获取机器人实例
    robot = scene["robot"]
    
    # 创建UI面板
    pose_monitor = LeapHandPoseMonitor(robot)     # 位姿监控面板
    control_panel = LeapHandControlPanel(robot)   # 关节控制面板
    
    # 仿真参数
    sim_dt = sim.get_physics_dt()
    count = 0
    
    print("\n" + "=" * 80)
    print("LeapHand 场景已启动")
    print(f"  - 监控的Body数量: {len(robot.data.body_names)}")
    print(f"  - 可控制的关节数: {len(robot.joint_names)}")
    print("  - UI面板:")
    print("    * LeapHand Body Pose Monitor (右侧): 实时位姿监控")
    print("    * LeapHand Joint Control (左侧): 关节角度控制")
    print("=" * 80 + "\n")
    
    # 仿真循环
    while simulation_app.is_running():
        # 将关节目标位置写入仿真 (关键步骤!)
        scene.write_data_to_sim()
        
        # 执行仿真步进
        sim.step()
        
        # 更新场景
        scene.update(sim_dt)
        
        # 更新位姿监控面板 (每10帧更新一次UI以提升性能)
        if count % 10 == 0:
            pose_monitor.update()
        
        count += 1


def main():
    """主函数"""
    # 创建仿真上下文
    sim_cfg = SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # 设置相机视角
    sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.5])
    
    # 创建场景
    scene_cfg = LeapHandSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 初始化仿真
    sim.reset()
    
    print("[INFO]: 场景设置完成，开始运行...")
    
    # 运行仿真器
    run_simulator(sim, scene)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()