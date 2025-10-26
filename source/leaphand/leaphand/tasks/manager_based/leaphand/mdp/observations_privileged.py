"""特权观测（Privileged Observations）

提供仅仿真可用的特权信息，如力矩等
提供在训练中常被随机化的物理与执行器参数（如关节刚度/阻尼、关节摩擦/电枢、刚体质量）的紧凑型统计作为 Critic 的输入。

NOTE:
        - 数值表示：除“材质摩擦/恢复”外，其余均以“当前值/默认值”的缩放比作为输入；缩放比更能直接暴露域随机化的尺度因子。
        - 统计聚合：对逐关节参数做 [mean, std] 聚合，得到稳定的小维度输入，避免与关节数量强绑定。
        - 安全回退：若底层API缺失，返回0以保持训练健壮性。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

###
# -特权信息
###


def goal_quat_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "goal_pose",
    make_quat_unique: bool = True,
) -> torch.Tensor:
    """物体当前姿态与目标姿态的四元数差（方向误差）。

    Args:
        env: ManagerBasedRLEnv - 环境实例
        asset_cfg: SceneEntityCfg - 物体资产配置
        command_name: str - 命令项名称（用于获取目标姿态）
        make_quat_unique: bool - 是否对四元数进行归一化处理

    Returns:
        (num_envs, 4) 张量，四元数形式的姿态差（从当前姿态到目标姿态的旋转）

    NOTE:
        - 计算公式：quat_diff = quat_target ⊗ quat_current^(-1)
        - 该四元数表示"从当前姿态旋转到目标姿态所需的旋转"
        - 如果 make_quat_unique=True，确保 w 分量非负（去除符号歧义）
    """
    import isaaclab.utils.math as math_utils

    # 获取物体当前姿态
    obj: RigidObject = env.scene[asset_cfg.name]
    current_quat = obj.data.root_quat_w  # (num_envs, 4) in (w, x, y, z)

    # 获取目标姿态（从命令管理器）
    # goal_pose 通常是 (pos, quat)，我们取后4维作为目标四元数
    goal_pose = env.command_manager.get_command(command_name)
    target_quat = goal_pose[:, -4:]  # (num_envs, 4) in (w, x, y, z)

    # 计算四元数差：quat_diff = target ⊗ current^(-1)
    current_quat_inv = math_utils.quat_inv(current_quat)
    quat_diff = math_utils.quat_mul(target_quat, current_quat_inv)

    # 归一化处理（确保 w 分量非负）
    if make_quat_unique:
        quat_diff = math_utils.quat_unique(quat_diff)

    return quat_diff
