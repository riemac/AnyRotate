#!/usr/bin/env python3
"""
LeapHand单指/多指正弦轨迹追踪诊断脚本

核心机制说明:
=============

1. set_joint_position_target() 工作机制：
   - 用户调用 robot.set_joint_position_target(q_des) 将目标位置写入 robot.data.joint_pos_target buffer
   - 调用 robot.write_data_to_sim() 后，Articulation._apply_actuator_model() 被触发
   - ImplicitActuator.compute() 计算 PD 输出:
       tau = stiffness * (q_des - q_actual) + damping * (dq_des - dq_actual) + tau_ff
   - 这些计算结果被设置进 PhysX（通过 set_dof_position_targets）
   - PhysX 每个物理步长都使用 same cached q_des，直到下一次 set_joint_position_target 更新

2. 动作频率与 Decimation:
    - 环境步长 dt_env = sim_dt * decimation
    - 每个 env_step 只更新一次目标动作，并在内部执行 decimation 个 sim_step
    - 同一个 q_des 在 decimation 个物理步长内持续有效
    - 更高的 decimation => q_des 持续更长物理时间 => 轨迹离散化程度增加

3. 关键参数影响：
   - stiffness 越高 => PD 增益更强 => 追踪误差减小，但力量可能饱和
   - damping 越高 => 阻尼更强 => 运动更缓（高阻尼可防止过冲）
   - decimation 越大 => 每个命令持续更长 => 轨迹离散化程度增加

多指测试:
    --joint-group index|middle|little|thumb|all  或使用 --joint-names 指定任意关节

用法:
    python test_actuator.py --amplitude 0.3 --frequency 1.0 --stiffness 5.0 --damping 0.5 --joint-group index
"""

import argparse
from pathlib import Path

# ============================================================================
# 必须在导入其他 IsaacLab/IsaacSim 模块前启动应用
# ============================================================================
from isaaclab.app import AppLauncher

# 解析命令行参数
parser = argparse.ArgumentParser(description="LeapHand单指/多指正弦轨迹追踪诊断工具")
AppLauncher.add_app_launcher_args(parser)

# 诊断参数
parser.add_argument(
    "--amplitude", type=float, default=0.3,
    help="正弦轨迹振幅（弧度），默认0.3"
)
parser.add_argument(
    "--frequency", type=float, default=1.0,
    help="正弦轨迹频率（Hz），默认1.0"
)
parser.add_argument(
    "--stiffness", type=float, default=3.0,
    help="执行器刚度（N/m），默认3.0"
)
parser.add_argument(
    "--damping", type=float, default=0.1,
    help="执行器阻尼（N·s/m），默认0.1"
)
parser.add_argument(
    "--duration", type=float, default=10.0,
    help="仿真运行时长（秒），默认10.0"
)
parser.add_argument(
    "--decimation", type=int, default=4,
    help="动作频率倍数（decimation），默认4"
)
parser.add_argument(
    "--joint-group", type=str, default="index",
    choices=["index", "middle", "little", "thumb", "all"],
    help="预设的手指关节组，默认 index (a_0~a_3)"
)
parser.add_argument(
    "--joint-names", type=str, default="",
    help="逗号分隔的关节名称列表（覆盖 joint-group），例如 'a_0,a_1'"
)

args = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ============================================================================
# 现在可以安全导入 IsaacLab 和其他 Omniverse 模块
# ============================================================================
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from leaphand.robots.leap import LEAP_HAND_CFG

FINGER_JOINT_GROUPS = {
    "index": ["a_0", "a_1", "a_2", "a_3"],
    "middle": ["a_4", "a_5", "a_6", "a_7"],
    "little": ["a_8", "a_9", "a_10", "a_11"],
    "thumb": ["a_12", "a_13", "a_14", "a_15"],
    "all": [
        "a_0", "a_1", "a_2", "a_3",
        "a_4", "a_5", "a_6", "a_7",
        "a_8", "a_9", "a_10", "a_11",
        "a_12", "a_13", "a_14", "a_15",
    ],
}


# ============================================================================
# 配置
# ============================================================================

@configclass
class LeapHandSceneCfg(InteractiveSceneCfg):
    """LeapHand单指诊断场景配置"""
    
    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    
    # 光源
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


class SineTrajectoryLogger:
    """记录和分析正弦轨迹追踪性能的数据容器"""
    
    def __init__(self, num_steps: int, num_joints: int, device: str = "cpu"):
        """初始化日志记录器
        
        Args:
            num_steps: 总仿真步数
            num_joints: 关节数（通常只关注单个关节）
            device: 计算设备
        """
        self.num_steps = num_steps
        self.num_joints = num_joints
        self.device = device
        
        # 记录缓冲区
        self.q_target = torch.zeros(num_steps, num_joints, device=device)    # 目标位置
        self.q_actual = torch.zeros(num_steps, num_joints, device=device)    # 实际位置
        self.error = torch.zeros(num_steps, num_joints, device=device)       # 追踪误差
        self.tau_computed = torch.zeros(num_steps, num_joints, device=device) # 计算力矩
        
        self.step_count = 0
    
    def record(self, q_target: torch.Tensor, q_actual: torch.Tensor, tau: torch.Tensor):
        """记录单个步长的数据"""
        if self.step_count < self.num_steps:
            self.q_target[self.step_count] = q_target[0].clone()  # 第一个环境
            self.q_actual[self.step_count] = q_actual[0].clone()
            self.tau_computed[self.step_count] = tau[0].clone()
            self.error[self.step_count] = (q_target - q_actual)[0].clone()
            self.step_count += 1
    
    def compute_metrics(self) -> dict:
        """计算轨迹追踪性能指标"""
        metrics = {}
        effective_steps = max(1, self.step_count)
        for joint_idx in range(self.num_joints):
            error = self.error[:effective_steps, joint_idx]
            metrics[f"joint_{joint_idx}"] = {
                "rms_error": torch.sqrt(torch.mean(error ** 2)).item(),
                "max_error": torch.max(torch.abs(error)).item(),
                "mean_error": torch.mean(torch.abs(error)).item(),
            }
        return metrics


# ============================================================================
# 主函数
# ============================================================================

def run_simulator(
    sim: SimulationContext,
    scene: InteractiveScene,
    args: argparse.Namespace,
    simulation_app,
):
    """运行仿真循环，执行多关节正弦轨迹追踪诊断"""

    robot = scene["robot"]
    sim_dt = sim.cfg.dt
    decimation = max(1, args.decimation)
    env_dt = sim_dt * decimation

    # 解析目标关节
    all_joint_names = list(robot.joint_names)
    if not all_joint_names:
        raise RuntimeError("无法获取机器人关节名称，请确认场景已正确初始化")

    if args.joint_names:
        candidate_names = [name.strip() for name in args.joint_names.split(",") if name.strip()]
    else:
        candidate_names = FINGER_JOINT_GROUPS.get(args.joint_group, ["a_0"])

    target_joint_names = []
    for name in candidate_names:
        if name in all_joint_names:
            target_joint_names.append(name)
        else:
            print(f"[WARNING] 关节 {name} 不存在，已忽略")

    if not target_joint_names:
        target_joint_names = [all_joint_names[0]]
        print(f"[WARNING] 使用默认关节 {target_joint_names[0]}")

    target_joint_ids = [all_joint_names.index(name) for name in target_joint_names]
    num_joints = len(target_joint_ids)

    print(f"[INFO] 目标关节: {target_joint_names}")

    # 参数
    amplitude = args.amplitude
    frequency = args.frequency
    stiffness = args.stiffness
    damping = args.damping
    duration = args.duration

    device = robot.device

    stiffness_tensor = torch.tensor(stiffness, device=device)
    damping_tensor = torch.tensor(damping, device=device)

    for joint_id in target_joint_ids:
        robot.write_joint_stiffness_to_sim(stiffness_tensor, joint_ids=[joint_id])
        robot.write_joint_damping_to_sim(damping_tensor, joint_ids=[joint_id])

    num_env_steps = max(1, int(math.ceil(duration / env_dt)))
    logger = SineTrajectoryLogger(num_env_steps, num_joints, device=device)

    q_home = robot.data.default_joint_pos[0, target_joint_ids].clone()

    print(f"\n[INFO] 仿真参数:")
    print(f"  - 振幅: {amplitude:.4f} rad")
    print(f"  - 频率: {frequency:.2f} Hz")
    print(f"  - 刚度: {stiffness:.2f} N/m")
    print(f"  - 阻尼: {damping:.4f} N·s/m")
    print(f"  - Decimation: {decimation}")
    print(f"  - 环境步长 dt: {env_dt:.4f} s")
    print(f"  - 运行时长: {duration:.2f} s (~{num_env_steps} env steps)")
    print(f"  - Home 位置: {q_home.tolist()}\n")

    target_pos_template = robot.data.default_joint_pos.clone()
    sim_time = 0.0

    for step in range(num_env_steps):
        if not simulation_app.is_running():
            break

        env_time = step * env_dt
        target_pos = target_pos_template.clone()
        q_targets = []
        for idx, joint_id in enumerate(target_joint_ids):
            q_des = q_home[idx].item() + amplitude * math.sin(2.0 * math.pi * frequency * env_time)
            q_targets.append(q_des)
            target_pos[0, joint_id] = q_des

        q_target_tensor = torch.tensor([q_targets], dtype=torch.float32, device=device)
        robot.set_joint_position_target(target_pos)
        robot.write_data_to_sim()

        for _ in range(decimation):
            sim.step()
            robot.update(sim_dt)
            sim_time += sim_dt

        q_actual = robot.data.joint_pos[0:1, target_joint_ids]
        tau = robot.data.applied_torque[0:1, target_joint_ids]

        logger.record(q_target_tensor, q_actual, tau)

        if step % max(1, num_env_steps // 10) == 0:
            errors = (q_target_tensor - q_actual).squeeze(0)
            error_str = ", ".join(
                f"{name}: {errors[idx].item():+.4f} rad"
                for idx, name in enumerate(target_joint_names)
            )
            print(
                f"[Step {step:5d}] env_t={env_time:.4f}s | targets={q_target_tensor.squeeze(0).tolist()} |"
                f" errors=({error_str})"
            )

    return logger, {
        "target_joint_names": target_joint_names,
        "amplitude": amplitude,
        "frequency": frequency,
        "stiffness": stiffness,
        "damping": damping,
        "decimation": decimation,
        "env_dt": env_dt,
        "joint_label": args.joint_names if args.joint_names else args.joint_group,
    }


def plot_results(logger: SineTrajectoryLogger, metadata: dict, output_dir: Path):
    """绘制追踪性能可视化"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    effective_steps = logger.step_count
    if effective_steps == 0:
        raise RuntimeError("没有可用的日志数据用于绘图")

    q_target = logger.q_target[:effective_steps].cpu().numpy()
    q_actual = logger.q_actual[:effective_steps].cpu().numpy()
    error = logger.error[:effective_steps].cpu().numpy()
    
    time = torch.arange(effective_steps, dtype=torch.float32).cpu().numpy() * metadata["env_dt"]
    joint_names = metadata["target_joint_names"]
    
    metrics = logger.compute_metrics()
    
    num_joints = len(joint_names)
    rel_denominator = np.clip(np.abs(q_target), 1e-5, None)
    rel_error = error / rel_denominator

    fig, axes = plt.subplots(3, num_joints, figsize=(max(10, 4 * num_joints), 11), sharex='row')
    if num_joints == 1:
        axes = axes.reshape(3, 1)

    fig.suptitle(
        f"LeapHand Sine Trajectory Tracking | Joints: {', '.join(joint_names)}\n"
        f"f={metadata['frequency']:.2f}Hz, A={metadata['amplitude']:.3f}rad, "
        f"K={metadata['stiffness']:.2f}N/m, C={metadata['damping']:.4f}N·s/m, Decimation={metadata['decimation']}",
        fontsize=13, fontweight='bold'
    )

    metrics_lines_abs = []
    metrics_lines_rel = []
    for idx, joint_name in enumerate(joint_names):
        ax_pos = axes[0, idx]
        ax_err = axes[1, idx]
        ax_rel = axes[2, idx]

        ax_pos.plot(time, q_target[:, idx], linewidth=2, label='Target')
        ax_pos.plot(time, q_actual[:, idx], linestyle="--", linewidth=1.5, label='Actual')
        ax_pos.set_title(joint_name, fontsize=11)
        ax_pos.grid(True, alpha=0.3)
        if idx == 0:
            ax_pos.set_ylabel('Position (rad)')
        ax_pos.legend(loc='upper right', fontsize=8)

        ax_err.plot(time, error[:, idx], linewidth=1.5, color='tab:green', label='Error')
        ax_err.grid(True, alpha=0.3)
        ax_err.set_xlabel('Time (s)')
        if idx == 0:
            ax_err.set_ylabel('Error (rad)')

        rel_series = rel_error[:, idx]
        ax_rel.plot(time, rel_series, linewidth=1.5, color='tab:purple', label='Rel Error')
        ax_rel.grid(True, alpha=0.3)
        ax_rel.set_xlabel('Time (s)')
        if idx == 0:
            ax_rel.set_ylabel('Rel. Error')

        m = metrics[f"joint_{idx}"]
        metrics_lines_abs.append(
            f"{joint_name}: RMS={m['rms_error']:.4f} rad | Max={m['max_error']:.4f} rad | Avr={m['mean_error']:.4f} rad"
        )
        ax_err.text(
            0.02, 0.95,
            f"RMS={m['rms_error']:.4f}\nMax={m['max_error']:.4f}\nAvr={m['mean_error']:.4f}",
            transform=ax_err.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        )

        rel_abs = np.abs(rel_series)
        rel_rms = float(np.sqrt(np.mean(rel_abs ** 2)))
        rel_max = float(np.max(rel_abs))
        rel_mean = float(np.mean(rel_abs))
        metrics_lines_rel.append(
            f"{joint_name}: RelRMS={rel_rms:.4f} | RelMax={rel_max:.4f} | RelAvr={rel_mean:.4f}"
        )
        ax_rel.text(
            0.02, 0.95,
            f"RelRMS={rel_rms:.4f}\nRelMax={rel_max:.4f}\nRelAvr={rel_mean:.4f}",
            transform=ax_rel.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#d4c2fc', alpha=0.7)
        )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    joint_label = metadata.get("joint_label", "custom").replace(",", "-")
    output_path = output_dir / (
        f"leaphand_sine_tracking_{joint_label}_f{metadata['frequency']:.1f}Hz_"
        f"K{metadata['stiffness']:.1f}_C{metadata['damping']:.3f}.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[INFO] 图表已保存: {output_path}")
    
    print("\n" + "="*60)
    print("追踪性能报告")
    print("="*60)
    print(f"关节: {', '.join(joint_names)}")
    print(f"频率: {metadata['frequency']:.2f} Hz | 振幅: {metadata['amplitude']:.3f} rad")
    print(f"刚度: {metadata['stiffness']:.2f} N/m | 阻尼: {metadata['damping']:.4f} N·s/m")
    print("绝对误差:")
    for line in metrics_lines_abs:
        print(f"  - {line}")
    print("相对误差:")
    for line in metrics_lines_rel:
        print(f"  - {line}")
    print("="*60 + "\n")
    
    return output_path


def main():
    """主函数"""
    
    # 仿真配置
    # 注意: decimation 是环境参数，而非仿真参数
    # env_dt = physics_dt × decimation
    sim_cfg = SimulationCfg(
        dt=0.001,  # 物理步长 1ms
        render_interval=args.decimation,  # 渲染间隔
        gravity=(0.0, 0.0, -9.81),
    )
    
    # 场景配置，单个环境
    scene_cfg = LeapHandSceneCfg(
        num_envs=1,  # 单环境
        env_spacing=2.5,
        replicate_physics=False,
    )
    
    # 在场景中添加 LeapHand 机器人
    scene_cfg.robot = LEAP_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=0.5,
                velocity_limit=100.0,
                stiffness=args.stiffness,
                damping=args.damping,
                friction=0.01,
                armature=0.001,
            ),
        },
    )
    
    # 创建仿真上下文
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.set_camera_view(eye=[1.0, 1.0, 0.5], target=[0.0, 0.0, 0.0])
    
    # 重置仿真以初始化所有资源
    sim.reset()
    
    # 运行仿真
    logger, metadata = run_simulator(sim, scene, args, simulation_app)
    
    # 绘制结果
    output_dir = Path("outputs") / "actuator_diagnostics"
    plot_results(logger, metadata, output_dir)
    


if __name__ == "__main__":
    main()
    simulation_app.close()