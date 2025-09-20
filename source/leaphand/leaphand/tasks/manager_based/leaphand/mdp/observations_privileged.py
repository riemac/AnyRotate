"""域随机化的特权观测（Privileged Observations）

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


def _safe_ratio(curr: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
    """逐元素缩放比，零安全处理。

    NOTE:
        - 公式: ratio = curr / clamp(default, != 0)
        - 若 default 为 0，则该位 ratio = 0 作为无信息回退。
    """
    eps_mask = default == 0
    safe_default = torch.where(eps_mask, torch.ones_like(default), default)
    ratio = curr / safe_default
    # where default was zero, fall back to zero (no information)
    ratio = torch.where(eps_mask, torch.zeros_like(ratio), ratio)
    return ratio


def _reduce_stats(x: torch.Tensor) -> torch.Tensor:
    """将最后一维聚合为 [mean, std] 统计，逐环境输出。

    NOTE:
        - 输入形状: (num_envs, N)
        - 输出形状: (num_envs, 2)
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    mean = torch.nan_to_num(torch.mean(x, dim=-1, keepdim=True), nan=0.0)
    std = torch.nan_to_num(torch.std(x, dim=-1, keepdim=True, unbiased=False), nan=0.0)
    return torch.cat([mean, std], dim=-1)


def robot_joint_stiffness_stats(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节刚度缩放比统计 [mean, std]

    NOTE:
        - 输入：当前关节刚度与默认刚度的比例
        - 输出形状: (num_envs, 2)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    curr = asset.data.joint_stiffness[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_stiffness[:, asset_cfg.joint_ids]
    ratio = _safe_ratio(curr, default)
    return _reduce_stats(ratio)


def robot_joint_damping_stats(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节阻尼缩放比统计 [mean, std]

    NOTE:
        - 输入：当前关节阻尼与默认阻尼的比例
        - 输出形状: (num_envs, 2)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    curr = asset.data.joint_damping[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_damping[:, asset_cfg.joint_ids]
    ratio = _safe_ratio(curr, default)
    return _reduce_stats(ratio)


def robot_joint_armature_stats(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节电枢缩放比统计 [mean, std]

    NOTE:
        - 输入：当前电枢与默认电枢的比例
        - 输出形状: (num_envs, 2)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # Armature buffers exist in ArticulationData; default_joint_armature is populated on build.
    curr = asset.data.joint_armature[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_armature[:, asset_cfg.joint_ids]
    ratio = _safe_ratio(curr, default)
    return _reduce_stats(ratio)


def robot_joint_friction_stats(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """关节摩擦系数缩放比统计 [mean, std]

    NOTE:
        - 输入：当前关节摩擦系数与默认值的比例
        - 输出形状: (num_envs, 2)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    curr = asset.data.joint_friction_coeff[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_friction_coeff[:, asset_cfg.joint_ids]
    ratio = _safe_ratio(curr, default)
    return _reduce_stats(ratio)


def object_total_mass_scale(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """物体总质量的缩放比（当前/默认）

    NOTE:
        - 输出形状: (num_envs, 1)
        - 首次调用缓存默认总质量，确保缩放比稳定；若 API 不可用则回退为 0。
    """
    obj: RigidObject = env.scene[object_cfg.name]
    device = env.device

    # Get per-env masses (sum over rigid bodies if needed)
    masses = None
    if hasattr(obj, "root_physx_view") and hasattr(obj.root_physx_view, "get_masses"):
        masses_tensor = obj.root_physx_view.get_masses().to(device)
        # masses may be (num_envs, num_bodies) or (num_envs,) depending on backend
        if masses_tensor.ndim == 1:
            masses = masses_tensor.unsqueeze(-1)
        else:
            masses = masses_tensor.sum(dim=-1, keepdim=True)
    # Fallback: use zeros if API is absent
    if masses is None:
        masses = torch.zeros((env.num_envs, 1), device=device)

    # Cache default masses once (first call), stored on env for stability
    cache_name = f"_{object_cfg.name}_default_total_mass"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, masses.clone().detach())
    default_masses = getattr(env, cache_name)

    ratio = _safe_ratio(masses, default_masses)
    return ratio

def object_material_friction_restitution(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """返回物体材质的原始物理参数 [static_friction, dynamic_friction, restitution]

    Args:
        env: ManagerBasedRLEnv - 环境实例
        object_cfg: SceneEntityCfg - 物体资产配置

    Returns:
        (num_envs, 3) 张量，分别为 [静摩擦, 动摩擦, 恢复系数]

    NOTE:
        - 用户要求材质摩擦使用原始值（[0,1] 范围），不做缩放比。
        - 若底层接口不可用，安全回退为 0 值。
        - 该观测仅应作为 critic 特权信息使用。
    """
    obj: RigidObject = env.scene[object_cfg.name]
    device = env.device

    # 统一形状为 (num_envs, 1) 的列向量，缺失时回退为 0
    static_friction = torch.zeros((env.num_envs, 1), device=device)
    dynamic_friction = torch.zeros((env.num_envs, 1), device=device)
    restitution = torch.zeros((env.num_envs, 1), device=device)

    # 通过 PhysX 视图读取（若可用）
    view = getattr(obj, "root_physx_view", None)
    if view is not None:
        # 尝试常见字段/方法名（不同版本可能不同）；失败则保持零回退
        # 不同版本API可能返回 (num_envs,) 或 (num_envs, 1)
        if hasattr(view, "get_material_static_friction"):
            try:
                v = view.get_material_static_friction().to(device)
                static_friction = v.unsqueeze(-1) if v.ndim == 1 else v
            except Exception:
                pass
        if hasattr(view, "get_material_dynamic_friction"):
            try:
                v = view.get_material_dynamic_friction().to(device)
                dynamic_friction = v.unsqueeze(-1) if v.ndim == 1 else v
            except Exception:
                pass
        if hasattr(view, "get_material_restitution"):
            try:
                v = view.get_material_restitution().to(device)
                restitution = v.unsqueeze(-1) if v.ndim == 1 else v
            except Exception:
                pass

    return torch.cat([static_friction, dynamic_friction, restitution], dim=-1)


def object_scale_ratio(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """物体统一尺度的缩放比（当前/默认），标量输出。

    Returns:
        (num_envs, 1) 张量，值为当前尺度相对于默认尺度的比例。

    NOTE:
        - 若底层接口提供逐轴缩放 (sx, sy, sz)，则取其均值作为统一尺度以稳定维度。
        - 首次调用缓存默认尺度，确保缩放比随环境重置稳定；若接口缺失则回退为 0。
        - rigid_body_scale 的域随机化建议使用 prestartup 模式（用户已按最佳实践设置）。
    """
    obj: RigidObject = env.scene[object_cfg.name]
    device = env.device

    # 读取当前尺度（均值聚合成标量），接口缺失则回退为0
    scale = torch.zeros((env.num_envs, 1), device=device)
    view = getattr(obj, "root_physx_view", None)
    if view is not None:
        try:
            if hasattr(view, "get_scales"):
                s = view.get_scales().to(device)
            elif hasattr(view, "get_local_scales"):
                s = view.get_local_scales().to(device)
            else:
                s = None
            if s is not None:
                if s.ndim == 2:
                    scale = s.mean(dim=-1, keepdim=True)
                elif s.ndim == 1:
                    scale = s.unsqueeze(-1)
        except Exception:
            pass

    # 默认尺度优先来自配置（未随机化的工厂值），否则首次调用时缓存
    cache_name = f"_{object_cfg.name}_default_scale"
    if not hasattr(env, cache_name):
        default_from_cfg = None
        try:
            s_cfg = getattr(obj.cfg.spawn, "scale", None)
            if s_cfg is not None:
                if isinstance(s_cfg, (tuple, list)):
                    # 取均值作为统一尺度
                    default_from_cfg = torch.tensor([sum(s_cfg) / len(s_cfg)], device=device).repeat(env.num_envs, 1)
                elif isinstance(s_cfg, (int, float)):
                    default_from_cfg = torch.tensor([float(s_cfg)], device=device).repeat(env.num_envs, 1)
        except Exception:
            default_from_cfg = None
        setattr(env, cache_name, (default_from_cfg if default_from_cfg is not None else scale.clone().detach()))
    default_scale = getattr(env, cache_name)

    ratio = _safe_ratio(scale, default_scale)
    return ratio


def object_com_offset(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """物体质心（COM）相对于默认值的本地坐标偏移。

    Returns:
        (num_envs, 3) 张量，局部坐标系下的 COM 偏移向量：current_com_local - default_com_local。

    NOTE:
        - 该量是位移，不做缩放比；单位米。
        - 首次调用缓存默认 COM；若接口缺失则回退为 0。
        - 对于单刚体物体，该量可直接反映域随机化的 COM 扰动大小与方向。
    """
    obj: RigidObject = env.scene[object_cfg.name]
    device = env.device

    # 通过 RigidObjectData 在 body 坐标系读取 COM（形状: [num_envs, 1, 3]）
    try:
        com_local = obj.data.body_com_pos_b.to(device)
        if com_local.ndim == 3:
            com_local = com_local[:, 0, :]
    except Exception:
        com_local = torch.zeros((env.num_envs, 3), device=device)

    cache_name = f"_{object_cfg.name}_default_com_local"
    if not hasattr(env, cache_name):
        setattr(env, cache_name, com_local.clone().detach())
    default_com = getattr(env, cache_name)

    offset = com_local - default_com
    return offset

