# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand连续旋转任务环境实现"""

from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Scale tensor from [-1, 1] range to [lower, upper] range.

    Args:
        x: Input tensor in [-1, 1] range
        lower: Lower bound of target range
        upper: Upper bound of target range

    Returns:
        Scaled tensor in [lower, upper] range
    """
    return 0.5 * (x + 1.0) * (upper - lower) + lower


class LeaphandContinuousRotEnv(DirectRLEnv):
    """LeapHand连续旋转任务环境类
    
    直接继承DirectRLEnv，实现连续旋转任务而非目标导向的旋转任务。
    主要特点：
    1. 鼓励持续旋转而非达到特定角度
    2. 奖励函数基于旋转速度而非目标距离
    3. 无需目标可视化标记功能
    """
    cfg: LeaphandContinuousRotEnvCfg

    def __init__(self, cfg: LeaphandContinuousRotEnvCfg, render_mode: str | None = None, **kwargs):
        # 直接调用DirectRLEnv的初始化
        super().__init__(cfg, render_mode, **kwargs)

        # 从场景中获取 hand 和 object 的引用
        self.hand = self.scene.articulations["hand"]
        self.object = self.scene.rigid_objects["object"]

        # 手部关节数量
        self.num_hand_dofs = self.hand.num_joints

        # 位置目标缓冲区：当前目标、上一步目标
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # 可驱动关节索引列表（根据配置的关节名查找索引并排序）
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # 指尖对应的刚体索引列表（根据配置的 body 名称查找并排序）
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # 关节位置限制（下限和上限）
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # 物体在手中的目标位置 (使用USDA文件中的默认位置)
        self.target_object_pos = None

        # 连续旋转任务特定变量
        self.cumulative_rotation = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.last_object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.last_object_rot[:, 0] = 1.0  # 初始化为单位四元数

        # 旋转方向缓冲区 - 随机选择旋转轴
        self.rotation_axis = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # 添加时间步长属性
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # 在场景设置完成后初始化目标位置
        self._initialize_target_positions()

    def _initialize_target_positions(self):
        """初始化目标位置为手部附近的位置"""
        # 设置目标位置为手部附近，而不是物体的初始位置
        # 这样物体可以从高处下落到手部附近
        hand_base_pos = self.hand.data.root_pos_w.clone()
        # 设置目标位置在手部上方约10cm处
        target_offset = torch.tensor([0.0, 0.0, 0.1], device=self.device).repeat(self.num_envs, 1)
        self.target_object_pos = hand_base_pos + target_offset

        # 初始化旋转轴（连续旋转任务特有）
        self._reset_rotation_axis(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self):
        """设置仿真场景，包括地面、机器人和物体"""
        # Add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, -0.1),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        # Add the hand articulation
        self.scene.articulations["hand"] = Articulation(self.cfg.robot_cfg)

        # Add the object as a rigid body
        self.scene.rigid_objects["object"] = RigidObject(self.cfg.object_cfg)

        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 1. 保存动作
        self.actions = actions.clone()

        # 2. 将动作从[-1,1]缩放到关节限制范围内
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        # 3. 应用移动平均滤波以平滑动作 - new_target = α × current_target + (1-α) × previous_target
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )

        # 4. 确保目标位置在关节限制范围内
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        # 5. 更新上一步目标位置
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

    def _apply_action(self) -> None:
        # 将目标位置应用到机器人关节
        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        """获取环境观测状态
        
        Returns:
            dict: 包含策略网络观测数据
        """
        # 首先计算中间值，确保object_pos等属性可用
        self._compute_intermediate_values()

        # 根据配置选择观测类型
        if self.cfg.obs_type == "openai":
            # 使用简化观测空间(仅包含关键信息)
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            # 使用完整观测空间(包含所有状态信息)
            obs = self.compute_full_observations()
        else:
            raise ValueError(f"Unknown observations type: {self.cfg.obs_type}")

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """计算连续旋转任务的奖励"""
        (
            total_reward,
            rotation_velocity_reward,
            grasp_reward,
            stability_reward,
            action_penalty,
            fall_penalty,
        ) = compute_continuous_rotation_rewards(
            self.object_pos,
            self.object_rot,
            self.last_object_rot,
            self.target_object_pos,
            self.rotation_axis,
            self.hand.data.body_pos_w[:, self.finger_bodies],  # fingertip positions
            self.actions,
            self.cfg.rotation_velocity_reward_scale,
            self.cfg.grasp_reward_scale,
            self.cfg.stability_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.fall_penalty,
            self.cfg.fall_dist,
            self.dt,
        )

        # 更新上一帧的物体旋转
        self.last_object_rot[:] = self.object_rot.clone()

        # 更新累积旋转
        self._update_cumulative_rotation()

        # 连续旋转任务不需要重置目标，因为没有特定目标
        # 但我们可以定期更换旋转轴以增加多样性
        if self.cfg.change_rotation_axis_interval > 0:
            change_axis_mask = (self.episode_length_buf % self.cfg.change_rotation_axis_interval) == 0
            change_axis_env_ids = change_axis_mask.nonzero(as_tuple=False).squeeze(-1)
            if len(change_axis_env_ids) > 0:
                self._reset_rotation_axis(change_axis_env_ids)

        if "log" not in self.extras:
            self.extras["log"] = dict()
        
        # 记录连续旋转特定的指标
        self.extras["log"]["rotation_velocity_reward"] = rotation_velocity_reward.mean()
        self.extras["log"]["cumulative_rotation_x"] = self.cumulative_rotation[:, 0].mean()
        self.extras["log"]["cumulative_rotation_y"] = self.cumulative_rotation[:, 1].mean()
        self.extras["log"]["cumulative_rotation_z"] = self.cumulative_rotation[:, 2].mean()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """检查环境完成条件"""
        self._compute_intermediate_values()

        # 检查物体是否掉落
        object_dist = torch.norm(self.object_pos - self.target_object_pos, p=2, dim=-1)
        out_of_reach = object_dist >= self.cfg.fall_dist

        # 连续旋转任务没有成功条件，只有超时和掉落
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """重置指定环境"""
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        
        # 调用父类重置方法
        super()._reset_idx(env_ids)

        # 重置连续旋转特定的缓冲区
        self.cumulative_rotation[env_ids] = 0

        # 确保object_rot已经初始化
        self._compute_intermediate_values()
        self.last_object_rot[env_ids] = self.object_rot[env_ids].clone()

        # 重置旋转轴
        self._reset_rotation_axis(env_ids)

        # 重置目标缓冲区
        dof_pos = self.hand.data.joint_pos[env_ids]
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos

    def _compute_intermediate_values(self):
        """计算中间值，用于奖励和完成条件计算"""
        # 计算物体位置和旋转
        self.object_pos = self.object.data.root_pos_w
        self.object_rot = self.object.data.root_quat_w

    def _reset_rotation_axis(self, env_ids):
        """为指定环境重置旋转轴"""
        num_resets = len(env_ids)
        
        if self.cfg.rotation_axis_mode == "random":
            # 生成随机旋转轴
            rotation_axes = torch.randn((num_resets, 3), device=self.device)
            rotation_axes = rotation_axes / torch.norm(rotation_axes, dim=-1, keepdim=True)
        elif self.cfg.rotation_axis_mode == "z_axis":
            # 主要围绕Z轴旋转
            rotation_axes = torch.zeros((num_resets, 3), device=self.device)
            rotation_axes[:, 2] = 1.0
        elif self.cfg.rotation_axis_mode == "mixed":
            # 混合模式：70%概率Z轴，30%概率随机轴
            rotation_axes = torch.zeros((num_resets, 3), device=self.device)
            z_axis_mask = torch.rand(num_resets, device=self.device) < 0.7
            rotation_axes[z_axis_mask, 2] = 1.0
            
            random_mask = ~z_axis_mask
            if random_mask.sum() > 0:
                random_axes = torch.randn((random_mask.sum(), 3), device=self.device)
                random_axes = random_axes / torch.norm(random_axes, dim=-1, keepdim=True)
                rotation_axes[random_mask] = random_axes
        else:
            raise ValueError(f"Unknown rotation_axis_mode: {self.cfg.rotation_axis_mode}")

        # 添加轴噪声
        if self.cfg.rotation_axis_noise > 0:
            axis_noise = sample_uniform(
                -self.cfg.rotation_axis_noise,
                self.cfg.rotation_axis_noise,
                (num_resets, 3),
                device=self.device,
            )
            rotation_axes += axis_noise
            rotation_axes = rotation_axes / torch.norm(rotation_axes, dim=-1, keepdim=True)

        self.rotation_axis[env_ids] = rotation_axes

    def _update_cumulative_rotation(self):
        """更新累积旋转量"""
        # 计算当前帧相对于上一帧的旋转变化
        quat_diff = quat_mul(self.object_rot, quat_conjugate(self.last_object_rot))
        
        # 将四元数转换为轴角表示
        angle = 2.0 * torch.acos(torch.clamp(torch.abs(quat_diff[:, 0]), max=1.0))
        axis = quat_diff[:, 1:4]
        axis_norm = torch.norm(axis, dim=-1, keepdim=True)
        
        # 避免除零
        valid_rotation = axis_norm.squeeze(-1) > 1e-6
        axis[valid_rotation] = axis[valid_rotation] / axis_norm[valid_rotation]
        
        # 计算沿指定旋转轴的旋转量
        rotation_along_axis = angle.unsqueeze(-1) * axis
        projected_rotation = torch.sum(rotation_along_axis * self.rotation_axis, dim=-1)
        
        # 累积旋转（分别记录每个轴的旋转）
        self.cumulative_rotation += rotation_along_axis

    def compute_full_observations(self):
        """计算连续旋转任务的完整观测空间，移除目标旋转信息"""
        # 物体位置和旋转
        object_pos = self.object_pos
        object_rot = self.object_rot

        # 手指关节位置和速度
        hand_dof_pos = self.hand.data.joint_pos
        hand_dof_vel = self.hand.data.joint_vel

        # 指尖位置 (用于判断抓取状态)
        fingertip_positions = self.hand.data.body_pos_w[:, self.finger_bodies]

        # 物体相对于目标位置的位置差异
        object_relative_pos = object_pos - self.target_object_pos

        # 物体角速度 (用于判断旋转状态)
        object_ang_vel = self.object.data.root_ang_vel_w

        # 当前旋转轴信息（让智能体知道应该沿哪个轴旋转）
        rotation_axis = self.rotation_axis

        # 拼接所有观测（移除目标旋转）
        # 维度: 物体位置(3) + 物体旋转(4) + 手指关节位置(16) + 手指关节速度(16) + 指尖位置(12) + 物体相对位置(3) + 物体角速度(3) + 旋转轴(3) = 60
        obs = torch.cat([
            object_pos,                                    # 3
            object_rot,                                    # 4
            hand_dof_pos,                                  # 16
            hand_dof_vel,                                  # 16
            fingertip_positions.view(self.num_envs, -1),   # 12 (4 fingertips * 3 coords)
            object_relative_pos,                           # 3
            object_ang_vel,                                # 3
            rotation_axis,                                 # 3
        ], dim=-1)

        return obs

    def compute_reduced_observations(self):
        """简化观测空间 - 仅包含关键信息，移除目标旋转"""
        # 物体旋转
        object_rot = self.object_rot

        # 手指关节位置
        hand_dof_pos = self.hand.data.joint_pos

        # 当前旋转轴
        rotation_axis = self.rotation_axis

        obs = torch.cat([
            object_rot,      # 4
            hand_dof_pos,    # 16
            rotation_axis,   # 3
        ], dim=-1)         # 总计 23 维

        return obs


@torch.jit.script
def compute_continuous_rotation_rewards(
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    last_object_rot: torch.Tensor,
    target_object_pos: torch.Tensor,
    rotation_axis: torch.Tensor,
    fingertip_pos: torch.Tensor,
    actions: torch.Tensor,
    rotation_velocity_reward_scale: float,
    grasp_reward_scale: float,
    stability_reward_scale: float,
    action_penalty_scale: float,
    fall_penalty: float,
    fall_dist: float,
    dt: float,
):
    """计算连续旋转任务的奖励函数"""
    
    # 1. 旋转速度奖励 - 主要奖励，鼓励沿指定轴持续旋转
    quat_diff = quat_mul(object_rot, quat_conjugate(last_object_rot))
    angle = 2.0 * torch.acos(torch.clamp(torch.abs(quat_diff[:, 0]), max=1.0))
    axis = quat_diff[:, 1:4]
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    
    # 避免除零
    valid_rotation = axis_norm.squeeze(-1) > 1e-6
    axis = torch.where(valid_rotation.unsqueeze(-1), axis / axis_norm, torch.zeros_like(axis))
    
    # 计算沿指定旋转轴的角速度
    angular_velocity = angle / dt
    projected_velocity = torch.sum(axis * rotation_axis, dim=-1) * angular_velocity
    
    # 奖励正向旋转，轻微惩罚反向旋转
    rotation_velocity_reward = rotation_velocity_reward_scale * torch.clamp(projected_velocity, min=-0.1)

    # 2. 抓取奖励 - 鼓励保持物体在手中
    object_dist = torch.norm(object_pos - target_object_pos, p=2, dim=-1)
    grasp_reward = grasp_reward_scale * torch.exp(-10.0 * object_dist)

    # 3. 稳定性奖励 - 鼓励稳定的抓取，但不过度限制旋转
    fingertip_to_object = fingertip_pos - object_pos.unsqueeze(1)
    fingertip_distances = torch.norm(fingertip_to_object, p=2, dim=-1)
    avg_fingertip_dist = torch.mean(fingertip_distances, dim=-1)
    stability_reward = stability_reward_scale * torch.exp(-1.0 * torch.abs(avg_fingertip_dist - 0.05))

    # 4. 动作惩罚 - 鼓励平滑的动作
    action_penalty = action_penalty_scale * torch.sum(actions ** 2, dim=-1)

    # 5. 跌落惩罚 - 物体掉落时的惩罚
    fall_reward = torch.where(object_dist >= fall_dist, fall_penalty, 0.0)

    # 总奖励
    total_reward = rotation_velocity_reward + grasp_reward + stability_reward + action_penalty + fall_reward

    return total_reward, rotation_velocity_reward, grasp_reward, stability_reward, action_penalty, fall_reward
