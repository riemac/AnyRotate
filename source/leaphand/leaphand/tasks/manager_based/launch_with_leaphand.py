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
    """实时监控LeapHand的Body位姿和自定义标记点位姿的UI面板"""
    
    def __init__(self, robot, custom_prims: list[tuple[str, str]] | None = None):
        """初始化监控面板
        
        Args:
            robot: LeapHand机器人实例
            custom_prims: 需要额外监控的标记点列表，格式为 [(parent_body_name, marker_relative_path), ...]
                         例如 [("fingertip", "index_tip_head")]
        """
        self.robot = robot
        self.body_names = robot.data.body_names
        self.custom_prims = custom_prims or []
        
        # 获取USD Stage（用于计算初始偏移）
        from isaaclab.sim.utils import get_current_stage
        self.stage = get_current_stage()
        
        # 为每个自定义Prim计算并存储相对父体的固定偏移（只需计算一次）
        self.custom_prim_offsets = {}  # {(parent_name, marker_path): (offset_pos, offset_quat)}
        self._compute_custom_prim_offsets()
        
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
                    
                    # 如果有自定义Prim，添加分隔符和自定义区域
                    if self.custom_prims:
                        ui.Separator(height=2)
                        ui.Label("Custom Marker Prims", 
                                height=25, 
                                style={"font_size": 16, "color": 0xFFFFAA00})
                        for parent_body, marker_path in self.custom_prims:
                            self._create_custom_prim_row(parent_body, marker_path)
    
    def _compute_custom_prim_offsets(self):
        """计算所有自定义Prim相对其父刚体的固定偏移（只调用一次）"""
        from isaaclab.sim.utils import resolve_prim_pose
        import torch
        
        # 获取第一个环境的机器人根路径
        robot_root_path = self.robot.root_physx_view.prim_paths[0]
        
        for parent_body, marker_path in self.custom_prims:
            # 构建完整路径
            parent_prim_path = f"{robot_root_path}/{parent_body}"
            marker_prim_path = f"{parent_prim_path}/{marker_path}"
            
            # 获取USD Prim
            parent_prim = self.stage.GetPrimAtPath(parent_prim_path)
            marker_prim = self.stage.GetPrimAtPath(marker_prim_path)
            
            if parent_prim.IsValid() and marker_prim.IsValid():
                # 计算标记点相对父体的偏移（静态计算，不受仿真影响）
                offset_pos_tuple, offset_quat_tuple = resolve_prim_pose(marker_prim, ref_prim=parent_prim)
                
                # 转换为torch tensor
                offset_pos = torch.tensor(offset_pos_tuple, device=self.robot.device, dtype=torch.float32)
                offset_quat = torch.tensor(offset_quat_tuple, device=self.robot.device, dtype=torch.float32)
                
                # 存储偏移
                key = (parent_body, marker_path)
                self.custom_prim_offsets[key] = (offset_pos, offset_quat)
                
                print(f"[INFO] 计算标记点偏移: {parent_body}/{marker_path}")
                print(f"       offset_pos: {offset_pos.cpu().numpy()}")
                print(f"       offset_quat: {offset_quat.cpu().numpy()}")
            else:
                print(f"[WARNING] 无效的Prim路径: {parent_body}/{marker_path}")
    
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
    
    def _create_custom_prim_row(self, parent_body: str, marker_path: str):
        """为自定义Prim创建位姿显示行
        
        Args:
            parent_body: 父刚体名称
            marker_path: 标记点相对父体的路径
        """
        with ui.CollapsableFrame(
            title=f"[Marker] {parent_body}/{marker_path}",
            height=0,
            collapsed=False,  # 默认展开以便观察
            style={"color": 0xFFFFAA00}  # 使用橙色区分
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
                key = f"_custom_{parent_body}/{marker_path}"
                self.pose_labels[key] = {
                    "pos": pos_label,
                    "quat": quat_label,
                    "parent_body": parent_body,
                    "marker_path": marker_path
                }
    
    def update(self):
        """更新所有body和自定义标记点的位姿显示（每帧调用）"""
        # 1. 更新机器人刚体的位姿
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
        
        # 2. 更新自定义Prim的位姿（使用PhysX数据 + 固定偏移）
        from isaaclab.utils.math import combine_frame_transforms
        
        for parent_body, marker_path in self.custom_prims:
            key = f"_custom_{parent_body}/{marker_path}"
            offset_key = (parent_body, marker_path)
            
            if key in self.pose_labels and offset_key in self.custom_prim_offsets:
                # 获取父刚体的实时位姿（从Phy sX）
                try:
                    parent_idx = self.body_names.index(parent_body)
                    parent_pos_w = body_poses_w[0, parent_idx, :3]  # (3,)
                    parent_quat_w = body_poses_w[0, parent_idx, 3:7]  # (4,) wxyz
                    
                    # 获取预计算的偏移
                    offset_pos, offset_quat = self.custom_prim_offsets[offset_key]
                    
                    # 计算标记点的世界位姿 = 父体位姿 ⊕ 偏移
                    marker_pos_w, marker_quat_w = combine_frame_transforms(
                        parent_pos_w.unsqueeze(0), parent_quat_w.unsqueeze(0),
                        offset_pos.unsqueeze(0), offset_quat.unsqueeze(0)
                    )
                    
                    # 转换为numpy显示
                    pos = marker_pos_w[0].cpu().numpy()
                    quat = marker_quat_w[0].cpu().numpy()
                    
                    # 更新UI标签
                    pos_str = f"x: {pos[0]:7.4f}  y: {pos[1]:7.4f}  z: {pos[2]:7.4f}"
                    quat_str = f"w: {quat[0]:6.3f}  x: {quat[1]:6.3f}  y: {quat[2]:6.3f}  z: {quat[3]:6.3f}"
                    
                    self.pose_labels[key]["pos"].text = pos_str
                    self.pose_labels[key]["quat"].text = quat_str
                    
                except ValueError:
                    # 父体名称不存在
                    if not self.pose_labels[key]["pos"].text.startswith("["):
                        self.pose_labels[key]["pos"].text = f"[INVALID PARENT: {parent_body}]"
                        self.pose_labels[key]["quat"].text = ""


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
            "Index": ["a_0", "a_1", "a_2", "a_3"],
            "Middle": ["a_4", "a_5", "a_6", "a_7"],
            "Ring": ["a_8", "a_9", "a_10", "a_11"],
            "Thumb": ["a_12", "a_13", "a_14", "a_15"],
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
    # 定义需要监控的自定义Prim（标记点）
    # 格式: (parent_body_name, marker_relative_path)
    custom_marker_prims = [
        ("fingertip", "index_tip_head"),        # 食指指尖标记
        ("thumb_fingertip", "thumb_tip_head"),  # 拇指指尖标记
        ("fingertip_2", "middle_tip_head"),     # 中指指尖标记
        ("fingertip_3", "ring_tip_head"),       # 无名指指尖标记
    ]
    
    pose_monitor = LeapHandPoseMonitor(robot, custom_prims=custom_marker_prims)  # 位姿监控面板
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
        # 从仿真中读取最新状态并更新所有场景实体的内部缓冲区。如果不调用 update()，时间戳不更新，访问属性时会读取缓存的旧数据
        scene.update(sim_dt)  # 这个不更新机器人数据的话，UI面板会没有变动
        
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