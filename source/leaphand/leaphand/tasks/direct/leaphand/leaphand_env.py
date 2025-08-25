# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeapHand手内旋转任务环境实现"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .leaphand_env_cfg import LeaphandEnvCfg


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


class LeaphandEnv(DirectRLEnv):
    """LeapHand手内旋转任务环境类"""
    cfg: LeaphandEnvCfg

    def __init__(self, cfg: LeaphandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 从场景中获取 hand 和 object 的引用
        self.hand = self.scene.articulations["hand"]
        self.object = self.scene.rigid_objects["object"]

        # 手部关节数量
        self.num_hand_dofs = self.hand.num_joints

        # 位置目标缓冲区：当前目标、上一步目标、以及用于写入的目标
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
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

        # 手内旋转任务特定变量
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 物体在手中的目标位置 (使用USDA文件中的默认位置)
        # 这将在_setup_scene之后初始化
        self.target_object_pos = None

        # 目标旋转 - 初始化为随机旋转
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0  # 初始化为单位四元数

        # 初始物体旋转 - 用于计算旋转差异
        self.initial_object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.initial_object_rot[:, 0] = 1.0

        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # 旋转进度跟踪
        self.rotation_progress = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_rotation_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # unit tensors for rotation axis generation
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # 在场景设置完成后初始化目标位置
        self._initialize_target_positions()

    def _setup_scene(self):
        """设置仿真场景，包括地面、机器人和物体"""
        # Add ground plane
        spawn_ground_plane( # 并不存在于self.scene中，作为静态实体不需要交互
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, -0.1),
            orientation=(1.0, 0.0, 0.0, 0.0),  # 确保地面水平
        )

        # Add the hand articulation - 使用USDA场景中的机器人
        self.scene.articulations["hand"] = Articulation(self.cfg.robot_cfg)

        # Add the object as a rigid body - 创建物体
        self.scene.rigid_objects["object"] = RigidObject(self.cfg.object_cfg)

        # Clone and replicate environments
        self.scene.clone_environments(copy_from_source=False)

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg) # 并不存在于self.scene中，作为静态实体不需要交互

    def _initialize_target_positions(self):
        """初始化目标位置为手部附近的位置"""
        # 设置目标位置为手部附近，而不是物体的初始位置
        # 这样物体可以从高处下落到手部附近
        hand_base_pos = self.hand.data.root_pos_w.clone()
        # 设置目标位置在手部上方约10cm处
        target_offset = torch.tensor([0.0, 0.0, 0.1], device=self.device).repeat(self.num_envs, 1)
        self.target_object_pos = hand_base_pos + target_offset

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
            dict: 包含策略网络和评论家网络(如果使用非对称观测)所需的观测数据
        """
        # 首先计算中间值，确保object_pos等属性可用
        self._compute_intermediate_values()

        # 如果使用非对称观测，获取指尖力传感器数据
        if self.cfg.asymmetric_obs:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]

        # 根据配置选择观测类型
        if self.cfg.obs_type == "openai":
            # 使用简化观测空间(仅包含关键信息)
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            # 使用完整观测空间(包含所有状态信息)
            obs = self.compute_full_observations()
        else:
            raise ValueError(f"Unknown observations type: {self.cfg.obs_type}")

        # 如果使用非对称观测，计算完整状态信息
        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        # 构建观测字典
        observations = {"policy": obs}  # 策略网络使用标准观测
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}  # 评论家网络使用完整状态信息
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """计算手内旋转任务的奖励"""
        (
            total_reward,
            reset_goal_buf,
            successes,
            _,  # 忽略返回的success（重复）
        ) = compute_hand_rotation_rewards(
            self.object_pos,
            self.object_rot,
            self.target_object_pos,
            self.goal_rot,
            self.hand.data.body_pos_w[:, self.finger_bodies],  # fingertip positions
            self.actions,
            self.cfg.rotation_reward_scale,
            self.cfg.grasp_reward_scale,
            self.cfg.stability_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.fall_penalty,
            self.cfg.reach_goal_bonus,
            self.cfg.rotation_tolerance,
            self.cfg.fall_dist,
        )

        # 更新缓冲区
        self.reset_goal_buf[:] = reset_goal_buf
        self.successes[:] = successes

        # 手动更新连续成功计数
        self.consecutive_successes *= self.cfg.av_factor
        self.consecutive_successes += (1.0 - self.cfg.av_factor) * successes.float().mean()

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """检查环境完成条件"""
        self._compute_intermediate_values()

        # 检查物体是否掉落 - 基于距离目标位置的检测
        object_dist = torch.norm(self.object_pos - self.target_object_pos, p=2, dim=-1)
        out_of_reach = object_dist >= self.cfg.fall_dist

        if self.cfg.max_consecutive_success > 0:
            # 检查旋转任务是否成功
            rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            success_condition = rot_dist <= self.cfg.rotation_tolerance

            # 重置episode长度缓冲区当达到目标时
            self.episode_length_buf = torch.where(
                success_condition,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success
        else:
            max_success_reached = torch.zeros_like(out_of_reach, dtype=torch.bool)

        # 检查超时
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached

        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """重置指定环境，参考官方InHandManipulationEnv实现"""
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        # 重置物体状态
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # 设置全局物体位置（局部位置 + 位置噪声 + 环境原点）
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        # 保存初始旋转（用于生成目标旋转）
        self.initial_object_rot[env_ids] = object_default_state[:, 3:7].clone()

        # 重置速度为零
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # 重置手部状态
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * sample_uniform(
            -1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device
        )
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # Reset buffers
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        # Reset success tracking
        self.successes[env_ids] = 0
        self.reset_goal_buf[env_ids] = 0
        self.rotation_progress[env_ids] = 0
        self.last_rotation_dist[env_ids] = 0

        # 重置目标旋转
        self._reset_target_pose(env_ids)

    def _compute_intermediate_values(self):
        """计算中间值，用于奖励和完成条件计算"""
        # 计算物体位置和旋转
        self.object_pos = self.object.data.root_pos_w
        self.object_rot = self.object.data.root_quat_w

        # 更新旋转进度跟踪
        current_rot_dist = rotation_distance(self.object_rot, self.goal_rot)
        self.rotation_progress = torch.maximum(
            self.rotation_progress,
            self.last_rotation_dist - current_rot_dist
        )
        self.last_rotation_dist = current_rot_dist

    def compute_full_observations(self):
        """计算完整观测空间，专门为手内旋转任务设计"""
        # 物体位置和旋转
        object_pos = self.object_pos
        object_rot = self.object_rot

        # 目标旋转 (不包括位置，因为物体应该保持在手中)
        goal_rot = self.goal_rot

        # 手指关节位置和速度
        hand_dof_pos = self.hand.data.joint_pos
        hand_dof_vel = self.hand.data.joint_vel

        # 指尖位置 (用于判断抓取状态)
        fingertip_positions = self.hand.data.body_pos_w[:, self.finger_bodies]

        # 物体相对于目标位置的位置差异
        object_relative_pos = object_pos - self.target_object_pos

        # 物体角速度 (用于判断旋转稳定性)
        object_ang_vel = self.object.data.root_ang_vel_w

        # 拼接所有观测
        # 维度: 物体位置(3) + 物体旋转(4) + 目标旋转(4) + 手指关节位置(16) + 手指关节速度(16) + 指尖位置(12) + 物体相对位置(3) + 物体角速度(3) = 61
        obs = torch.cat([
            object_pos,                                    # 3
            object_rot,                                    # 4
            goal_rot,                                      # 4
            hand_dof_pos,                                  # 16
            hand_dof_vel,                                  # 16
            fingertip_positions.view(self.num_envs, -1),   # 12 (4 fingertips * 3 coords)
            object_relative_pos,                           # 3
            object_ang_vel,                                # 3
        ], dim=-1)

        return obs

    def compute_reduced_observations(self):
        """简化观测空间 - 仅包含关键信息"""
        # 物体旋转和目标旋转
        object_rot = self.object_rot
        goal_rot = self.goal_rot

        # 手指关节位置
        hand_dof_pos = self.hand.data.joint_pos

        obs = torch.cat([
            object_rot,      # 4
            goal_rot,        # 4
            hand_dof_pos     # 16
        ], dim=-1)         # 总计 24 维

        return obs

    def compute_full_state(self):
        # For asymmetric observation, include additional information
        return self.compute_full_observations()

    def _reset_target_pose(self, env_ids):
        """为指定环境重置目标旋转"""
        num_resets = len(env_ids)

        # 生成随机旋转轴
        rotation_axes = torch.randn((num_resets, 3), device=self.device)
        rotation_axes = rotation_axes / torch.norm(rotation_axes, dim=-1, keepdim=True)

        # 添加轴噪声
        axis_noise = sample_uniform(
            -self.cfg.rotation_axis_noise,
            self.cfg.rotation_axis_noise,
            (num_resets, 3),
            device=self.device,
        )
        rotation_axes += axis_noise
        rotation_axes = rotation_axes / torch.norm(rotation_axes, dim=-1, keepdim=True)

        # 生成随机旋转角度
        rotation_angles = sample_uniform(
            -self.cfg.target_rotation_range,
            self.cfg.target_rotation_range,
            (num_resets,),
            device=self.device,
        )

        # 从轴角表示转换为四元数
        target_rotations = quat_from_angle_axis(rotation_angles, rotation_axes)

        # 目标旋转 = 初始旋转 * 目标旋转变换
        self.goal_rot[env_ids] = quat_mul(self.initial_object_rot[env_ids], target_rotations)


@torch.jit.script
def compute_hand_rotation_rewards(
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_object_pos: torch.Tensor,
    goal_rot: torch.Tensor,
    fingertip_pos: torch.Tensor,
    actions: torch.Tensor,
    rotation_reward_scale: float,
    grasp_reward_scale: float,
    stability_reward_scale: float,
    action_penalty_scale: float,
    fall_penalty: float,
    reach_goal_bonus: float,
    rotation_tolerance: float,
    fall_dist: float,
):
    """计算手内旋转任务的奖励函数"""

    # 1. 旋转奖励 - 主要奖励，鼓励物体朝目标旋转
    # 计算旋转距离 (内联rotation_distance函数)
    quat_diff = quat_mul(object_rot, quat_conjugate(goal_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    rotation_reward = rotation_reward_scale * torch.exp(-5.0 * rot_dist)

    # 2. 抓取奖励 - 鼓励保持物体在手中
    object_dist = torch.norm(object_pos - target_object_pos, p=2, dim=-1)
    grasp_reward = grasp_reward_scale * torch.exp(-10.0 * object_dist)

    # 3. 稳定性奖励 - 鼓励平稳的旋转，惩罚剧烈运动
    # 计算指尖到物体的距离变化，鼓励稳定抓取
    fingertip_to_object = fingertip_pos - object_pos.unsqueeze(1)  # [num_envs, num_fingertips, 3]
    fingertip_distances = torch.norm(fingertip_to_object, p=2, dim=-1)  # [num_envs, num_fingertips]
    avg_fingertip_dist = torch.mean(fingertip_distances, dim=-1)  # [num_envs]
    stability_reward = stability_reward_scale * torch.exp(-2.0 * torch.abs(avg_fingertip_dist - 0.05))

    # 4. 动作惩罚 - 鼓励平滑的动作
    action_penalty = action_penalty_scale * torch.sum(actions ** 2, dim=-1)

    # 5. 跌落惩罚 - 物体掉落时的惩罚
    fall_reward = torch.where(object_dist >= fall_dist, fall_penalty, 0.0)

    # 6. 成功奖励 - 达到目标旋转时的奖励
    success = torch.where(rot_dist <= rotation_tolerance, torch.ones_like(rot_dist), torch.zeros_like(rot_dist))
    success_bonus = success * reach_goal_bonus

    # 总奖励
    total_reward = rotation_reward + grasp_reward + stability_reward + action_penalty + fall_reward + success_bonus

    # 重置目标缓冲区
    reset_goal_buf = (success > 0.5)

    return total_reward, reset_goal_buf, success, success


@torch.jit.script
def rotation_distance(quat1: torch.Tensor, quat2: torch.Tensor) -> torch.Tensor:
    quat_diff = quat_mul(quat1, quat_conjugate(quat2))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))


@torch.jit.script
def random_quaternions(n: int, device: torch.device) -> torch.Tensor:
    """
    Generate random quaternions
    """
    u = torch.rand(n, device=device)
    v = torch.rand(n, device=device)
    w = torch.rand(n, device=device)
    
    return torch.stack([
        torch.sqrt(1-u) * torch.sin(2 * np.pi * v),
        torch.sqrt(1-u) * torch.cos(2 * np.pi * v),
        torch.sqrt(u) * torch.sin(2 * np.pi * w),
        torch.sqrt(u) * torch.cos(2 * np.pi * w)
    ], dim=-1)