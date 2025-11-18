# Sim2Real 技术分析与实施方案

> **作者**: AI Agent  
> **日期**: 2025-11-18  
> **目的**: 分析 LEAP Hand 官方 sim2real 实现，对比 IsaacLab 标准 workflow，为 ManagerBasedRLEnv 环境提供 sim2real 适配方案

---

## 目录

1. [Leaphand 官方 ROS2 Sim2Real 适用范围](#1-leaphand-官方-ros2-sim2real-适用范围)
2. [您的环境与官方环境的差异](#2-您的环境与官方环境的差异)
3. [IsaacLab 官方 Sim2Real Workflow](#3-isaaclab-官方-sim2real-workflow)
4. [自定义接口的可行性与最佳实践](#4-自定义接口的可行性与最佳实践)
5. [推荐方案与实施步骤](#5-推荐方案与实施步骤)

---

## 1. Leaphand 官方 ROS2 Sim2Real 适用范围

### 1.1 适用性分析

**LEAP Hand 官方的 sim2real 实现主要针对 `DirectRLEnv` 架构下的 `reorientation_env.py` 环境**，具体表现为：

#### 环境架构特征
```python
# 官方环境: DirectRLEnv 架构
class ReorientationEnv(DirectRLEnv):
    def _get_observations(self) -> dict:
        # 直接拼接原始观测
        frame = unscale(self.hand_dof_pos, ...)
        if self.cfg.store_cur_actions:
            frame = torch.cat((frame, self.cur_targets[:]), dim=-1)
        
        # 手动管理历史缓存
        self.obs_hist_buf[:, :, :-1] = self.obs_hist_buf[:, :, 1:]
        self.obs_hist_buf[:, :, -1] = frame
        obs = self.obs_hist_buf.transpose(1, 2).reshape(self.num_envs, -1)
        return {"policy": obs.float()}
```

#### 硬编码的观测结构
- **观测维度**: 固定为 `num_proprio_obs = 16 * hist_len` (仅关节位置) 或 `32 * hist_len` (关节位置 + 动作目标)
- **无模块化**: 观测直接在 `_get_observations()` 中硬编码拼接，无法动态组合
- **特定任务**: 专为 Z 轴旋转任务设计，不包含物体姿态等任务相关观测

### 1.2 部署代码特点

#### `reorient_z_ros.py` 的核心流程
```python
class HardwarePlayer(Node):
    def __init__(self, cfg_path, ckpt_path):
        # 固定配置
        self.action_scale = 1 / 24
        self.action_type = "relative"
        self.actions_num = 16  # 硬编码关节数
        self.num_proprio_obs = 16  # 或 32
        self.hist_len = 3
        
    def poll_joint_position(self):
        """从真实机器人读取关节状态"""
        # ROS2 服务调用获取硬件状态
        response = self.cli.call_async(self.req)
        joint_position = self.real_to_sim(joint_position)  # 坐标转换
        return {'position': joint_position}
        
    def command_joint_position(self, desired_pose):
        """发送动作指令到真实机器人"""
        desired_pose = self.sim_to_real(desired_pose)  # 坐标转换
        joint_msg = JointState()
        joint_msg.position = desired_pose
        self.pub_hand.publish(joint_msg)
        
    def deploy(self):
        """主控制循环"""
        # 初始化历史缓存
        obs_hist_buf = torch.zeros((1, 32, self.hist_len), ...)
        
        while True:
            # 1. 策略推理
            action = self.forward_network(obs_buf)
            
            # 2. 动作执行
            target = prev_target + self.action_scale * action
            self.command_joint_position(target)
            
            # 3. 状态读取
            robot_state = self.poll_joint_position()
            
            # 4. 观测更新
            frame = torch.cat([unscaled_pos, target], dim=-1)
            obs_hist_buf[:, :, :-1] = obs_hist_buf[:, :, 1:]
            obs_hist_buf[:, :, -1] = frame
            obs_buf = obs_hist_buf.transpose(1, 2).reshape(1, -1)
```

### 1.3 适用范围结论

**❌ 官方 sim2real 代码不能直接适用于其他环境，原因如下**:

| 限制因素 | 具体表现 |
|---------|---------|
| **硬编码观测** | 仅支持 `joint_pos` + `last_action`，无法扩展到物体姿态、目标姿态等 |
| **固定架构** | 专为 DirectRLEnv 设计，无法直接对接 ManagerBasedRLEnv 的观测管理器 |
| **任务特定** | 仅适用于不需要物体状态反馈的任务（如盲操作） |
| **手动历史管理** | 需手动维护 `obs_hist_buf`，与 IsaacLab 的 `history_length` 机制不兼容 |

---

## 2. 您的环境与官方环境的差异

### 2.1 架构对比表

| 特性 | 官方 DirectRLEnv | 您的 ManagerBasedRLEnv |
|-----|-----------------|----------------------|
| **环境基类** | `DirectRLEnv` | `ManagerBasedRLEnv` |
| **观测定义** | `_get_observations()` 硬编码 | `ObservationsCfg` 声明式配置 |
| **动作定义** | `_apply_action()` 手动实现 | `ActionsCfg` 配置 + `ActionManager` |
| **奖励定义** | `_get_rewards()` 手动计算 | `RewardsCfg` 配置 + `RewardManager` |
| **历史管理** | 手动维护 `obs_hist_buf` | `history_length` 参数自动管理 |
| **域随机化** | `_reset_idx()` 中硬编码 | `EventCfg` 配置 + `EventManager` |
| **模块化** | 低（一切在环境类中） | 高（Manager 模式分离关注点） |

### 2.2 您的环境优势

#### 声明式观测配置
```python
@configclass
class ProprioceptionObsCfg(ObsGroup):
    """仅本体感受信息 - 可部署到真机"""
    joint_pos = ObsTerm(
        func=mdp.joint_pos_limit_normalized,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    goal_pose = ObsTerm(
        func=mdp.generated_commands, 
        params={"command_name": "goal_pose"}
    )
    last_action = ObsTerm(func=mdp.last_action)
    
    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True

@configclass
class PrivilegedObsCfg(ProprioceptionObsCfg):
    """包含特权信息 - 仅仿真可用"""
    object_pos = ObsTerm(func=mdp.root_pos_w, ...)
    object_quat = ObsTerm(func=mdp.root_quat_w, ...)
    goal_quat_diff = ObsTerm(func=leaphand_mdp.goal_quat_diff, ...)
```

**优势**:
- ✅ **观测模块化**: 每个观测项独立定义，易于添加/删除
- ✅ **Teacher-Student 分离**: 通过继承轻松定义不同观测组
- ✅ **自动历史**: `history_length=2` 自动管理时序信息
- ✅ **噪音注入**: 通过 `noise=Gnoise(std=0.002)` 模拟传感器噪音

#### 统一的动作接口
```python
@configclass
class ActionsCfg:
    hand_joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],
        scale=1.0,
        rescale_to_limits=True,
        alpha=1/24,  # 与官方的 action_scale 等价
    )
```

**优势**:
- ✅ **动作平滑**: EMA 机制内置，无需手动维护 `prev_target`
- ✅ **自动缩放**: `rescale_to_limits` 自动映射 [-1,1] → 关节限位
- ✅ **可配置**: 可轻松切换不同动作空间（位置/速度/力矩）

### 2.3 关键差异总结

```
┌─────────────────────────────────────────────────────────────┐
│              官方 DirectRLEnv 架构                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ _get_observations()                                  │   │
│  │   ├─ 硬编码 joint_pos                                │   │
│  │   ├─ 硬编码 last_action                              │   │
│  │   └─ 手动管理 obs_hist_buf                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│                  单一 Observation Vector                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│            您的 ManagerBasedRLEnv 架构                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ObservationManager                                   │   │
│  │   ├─ ProprioceptionObsCfg (可部署)                   │   │
│  │   │   ├─ joint_pos (ObsTerm)                         │   │
│  │   │   ├─ goal_pose (ObsTerm)                         │   │
│  │   │   └─ last_action (ObsTerm)                       │   │
│  │   │                                                   │   │
│  │   └─ PrivilegedObsCfg (仅仿真)                        │   │
│  │       ├─ object_pos (ObsTerm)                        │   │
│  │       ├─ object_quat (ObsTerm)                       │   │
│  │       └─ goal_quat_diff (ObsTerm)                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│            多个 Observation Groups (policy/critic)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. IsaacLab 官方 Sim2Real Workflow

### 3.1 标准三阶段 Teacher-Student 蒸馏

根据 IsaacLab 官方文档 (`sim-to-real.rst`)，标准 sim2real workflow 包含:

#### **阶段 1: 训练 Teacher Policy (带特权信息)**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-G1-v1 \
  --num_envs=4096 \
  --headless
```

**特征**:
- 使用 `PolicyCfg(ObsGroup)`，包含特权观测 (如 `root_linear_velocity`)
- 充分利用仿真优势，获得最优性能

#### **阶段 2: 蒸馏 Student Policy (仅真实传感器)**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Velocity-G1-Distillation-v1 \
  --num_envs=4096 \
  --load_run 2025-08-13_23-53-28 \
  --checkpoint model_1499.pt
```

**特征**:
- 使用 `StudentPolicyCfg(ObsGroup)`，仅包含真实传感器可测量项
- 通过行为克隆 (Behavior Cloning) 学习 teacher 策略:
  $$\mathcal{L}_{\text{BC}} = \mathbb{E}\left[ \| \pi_{\text{teacher}}(O_{\text{priv}}) - \pi_{\text{student}}(O_{\text{sensor}}) \|^2 \right]$$

#### **阶段 3: RL Fine-tune Student Policy**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Velocity-G1-Student-Finetune-v1 \
  --num_envs=4096 \
  --load_run 2025-08-20_16-06-52_distillation \
  --checkpoint model_1499.pt
```

**特征**:
- 从蒸馏后的 student policy 初始化
- 继续用 RL 优化，弥补蒸馏损失

### 3.2 观测组定义示例 (来自 velocity_env_cfg.py)

```python
@configclass
class PolicyCfg(ObsGroup):
    """Teacher Policy - 包含特权信息"""
    # 真实传感器可测量
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # IMU
    projected_gravity = ObsTerm(func=mdp.projected_gravity)  # IMU
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 编码器
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)  # 编码器
    actions = ObsTerm(func=mdp.last_action)  # 控制器输出
    
    # 特权信息（仿真专用）
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # ❌ 真机无法测量
    
@configclass
class StudentPolicyCfg(ObsGroup):
    """Student Policy - 仅真实传感器"""
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    actions = ObsTerm(func=mdp.last_action)
    # ❌ 移除 base_lin_vel
```

### 3.3 IsaacLab Sim2Real 的核心理念

| 原则 | 说明 |
|-----|------|
| **观测分离** | 通过不同 `ObsGroup` 区分 teacher 和 student 观测 |
| **模块化设计** | 每个观测项是独立的 `ObsTerm`，易于组合 |
| **蒸馏传递知识** | 让 student 在受限观测下学习 teacher 的行为 |
| **仿真验证** | 先 sim-to-sim 验证，再真机部署 |

---

## 4. 自定义接口的可行性与最佳实践

### 4.1 您的问题：是否可以自定义硬件接口？

**✅ 完全可行，且符合 IsaacLab 的 sim2real 理念！**

IsaacLab 的 `play.py` 本质上是一个 **策略推理循环**，与硬件无关。真机部署时需要:
1. **替换环境的 step() 函数** → 改为硬件读写接口
2. **保持策略推理逻辑不变** → 复用训练好的模型

### 4.2 参考架构：官方 Spot 四足机器人部署

IsaacLab 官方文档提到 [Spot Quadruped Deployment](https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/)，其架构为:

```
┌─────────────────────────────────────────────────────────────┐
│                 仿真训练 (IsaacLab)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ManagerBasedRLEnv                                    │   │
│  │   ├─ ObservationManager (读取仿真传感器)              │   │
│  │   ├─ ActionManager (发送仿真指令)                     │   │
│  │   └─ PolicyNetwork (RL 训练)                         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓ 导出模型
┌─────────────────────────────────────────────────────────────┐
│                 真机部署 (Custom Script)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ HardwareInterface (ROS/SDK)                          │   │
│  │   ├─ poll_sensor_data() → 读取 IMU/编码器            │   │
│  │   ├─ send_command() → 发送关节控制指令               │   │
│  │   └─ PolicyNetwork.forward() ← 加载导出模型          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**关键点**:
- ✅ **仿真训练** 使用完整的 ManagerBasedRLEnv
- ✅ **真机部署** 自定义轻量级硬件接口
- ✅ **策略复用** 加载导出的 `.pt` 或 `.onnx` 模型

### 4.3 您的环境如何适配？

#### 方案对比

| 方案 | 复用官方代码 | 工作量 | 灵活性 | 推荐度 |
|-----|-------------|-------|-------|-------|
| **A. 修改官方 reorient_z_ros.py** | ❌ 仅适用 DirectRLEnv | ⭐⭐⭐⭐ 高 | ⭐ 低 | ❌ 不推荐 |
| **B. 基于 play.py 封装硬件接口** | ✅ 复用 play.py 框架 | ⭐⭐⭐ 中 | ⭐⭐⭐ 中 | ⚠️ 可行但不优雅 |
| **C. 自定义 sim2real 脚本** | ⚠️ 参考官方思路 | ⭐⭐ 低 | ⭐⭐⭐⭐ 高 | ✅ **推荐** |

#### 推荐方案 C 的理由

1. **观测已模块化**: 您的 `ProprioceptionObsCfg` 已定义好部署观测
2. **动作已统一**: `EMAJointPositionToLimitsActionCfg` 封装动作处理逻辑
3. **模型易加载**: 使用官方的 `RLGamesPolicy` 类
4. **硬件独立**: 解耦仿真环境与硬件通信

---

## 5. 推荐方案与实施步骤

### 5.1 整体架构设计 (参考 play.py)

基于您的想法，我们采用以下目录结构：

```
scripts/rl_games/
├── play.py                    # 仿真策略回放（原有）
├── sim2real_play.py           # 真机部署主循环（新增）★
└── sim2real/                  # 真机部署模块（新增）
    ├── __init__.py
    ├── utils/                 # 硬件通信工具
    │   ├── __init__.py
    │   ├── ros2_interface.py  # ROS2 通信封装
    │   └── coordinate_transform.py  # 坐标系转换
    └── mdp/                   # MDP 组件（对齐仿真）
        ├── __init__.py
        ├── actions.py         # 动作处理（复现 ActionManager）
        ├── observations.py    # 观测处理（复现 ObservationManager）
        └── commands.py        # 命令管理（复现 CommandManager）
```

#### 核心设计理念

```
┌─────────────────────────────────────────────────────────────┐
│              仿真训练 (play.py)                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ gym.make(task_name, cfg=env_cfg)                     │   │
│  │   ↓                                                  │   │
│  │ ManagerBasedRLEnv                                    │   │
│  │   ├─ ObservationManager.compute_group("policy")     │   │
│  │   ├─ ActionManager.process_action(action)           │   │
│  │   └─ CommandManager.get_command("goal_pose")        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓ 架构对齐
┌─────────────────────────────────────────────────────────────┐
│          真机部署 (sim2real_play.py)                         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 不创建仿真环境，直接使用 MDP 组件                      │   │
│  │   ├─ observations.PolicyObservation(cfg)             │   │
│  │   ├─ actions.EMAJointPositionAction(cfg)             │   │
│  │   ├─ commands.ContinuousRotationCommand(cfg)         │   │
│  │   └─ utils.ROS2Interface(hardware_cfg)               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**关键思想**:
- ✅ **复用训练配置**: 读取 `inhand_base_env_cfg.py` 的 `ObservationsCfg/ActionsCfg/CommandsCfg`
- ✅ **对齐处理逻辑**: `sim2real/mdp/` 中的组件精确复现 IsaacLab 的 Manager 行为
- ✅ **解耦硬件通信**: `utils/ros2_interface.py` 负责与真机通信，其他代码无感知

### 5.2 已创建的代码框架

#### 目录结构（已完成）

```
scripts/rl_games/
├── play.py                              # 仿真策略回放（原有）
├── sim2real_play.py                     # ✅ 真机部署主脚本
└── sim2real/                            # ✅ Sim2Real 模块
    ├── __init__.py
    ├── utils/                           # 硬件通信工具
    │   ├── __init__.py
    │   ├── ros2_interface.py            # ✅ ROS2 硬件接口
    │   └── coordinate_transform.py      # ✅ 坐标系转换
    └── mdp/                             # MDP 组件（对齐仿真）
        ├── __init__.py
        ├── observations.py              # ✅ 观测处理
        ├── actions.py                   # ✅ 动作处理 (EMA)
        └── commands.py                  # ✅ 命令管理
```

#### 核心组件说明

| 文件 | 功能 | 对齐目标 |
|------|------|---------|
| `sim2real_play.py` | 主部署循环，参考 `play.py` 架构 | 替代 `gym.make()` + `env.step()` |
| `ros2_interface.py` | ROS2 硬件通信，包含坐标转换 | 替代 `env.step()` 的硬件读写 |
| `observations.py` | 观测拼接和历史管理 | 复现 `ObservationManager.compute_group()` |
| `actions.py` | EMA 动作平滑和限位映射 | 复现 `ActionManager.process_action()` |
| `commands.py` | 目标姿态管理和增量旋转 | 复现 `CommandManager.get_command()` |

### 5.3 分步实施计划

#### 步骤 1: 定义部署观测配置 ✅ **已完成**

您已经定义了 `ProprioceptionObsCfg`，需确认其仅包含真机可测量项:

```python
@configclass
class ProprioceptionObsCfg(PrivilegedObsCfg):
    """✅ 可部署到真机的观测"""
    # ✅ joint_pos: 关节编码器可测量
    # ✅ goal_pose: 通过命令管理器设定
    # ✅ last_action: 上一步策略输出
    
    # ❌ 移除仿真特权项
    object_pos = None
    object_quat = None
    goal_quat_diff = None
```

**验证清单**:
- [ ] 所有观测项都能从真实传感器获取
- [ ] 观测维度与训练时一致
- [ ] 噪音配置合理（真实传感器噪音 ≥ 仿真噪音）

#### 步骤 2: 验证已有模型或训练新模型

**选项 A: 验证已有模型** (如果模型使用了部署兼容的观测)
```bash
cd /home/hac/isaac/AnyRotate
source /home/hac/isaac/env_isaaclab/bin/activate

# 检查模型观测配置
python scripts/rl_games/play.py \
  --task Isaac-Leaphand-InHand-Object-Rot-v0 \
  --checkpoint logs/rl_games/leaphand_object_rot/2025-11-17_20-43-31/nn/checkpoint.pth \
  --num_envs 1
  
# 如果能正常运行且不依赖 object_pos/object_quat 等特权观测，则可直接部署
```

**选项 B: 训练新模型** (如果需要纯本体感受的策略)
```bash
# 修改 inhand_base_env_cfg.py:
# policy: ObsGroup = ProprioceptionObsCfg(history_length=2)

python scripts/rl_games/train.py \
  --task Isaac-Leaphand-InHand-Object-Rot-v0 \
  --num_envs 4096 \
  --headless
```

**选项 C: Teacher-Student 蒸馏** (推荐，如已有高性能 teacher)
```bash
# TODO: 参考 IsaacLab 官方蒸馏流程
# 1. 使用 PrivilegedObsCfg 训练 teacher
# 2. 蒸馏到 ProprioceptionObsCfg student
# 3. RL fine-tune student
```

#### 步骤 3: 测试 Sim2Real 代码逻辑 ✅ **框架已完成**

在连接真实硬件前，先用模拟接口测试代码逻辑：

```bash
cd /home/hac/isaac/AnyRotate
source /home/hac/isaac/env_isaaclab/bin/activate

# 使用模拟接口测试 (不需要真实硬件)
python scripts/rl_games/sim2real_play.py \
  --checkpoint logs/rl_games/leaphand_object_rot/2025-11-17_20-43-31/nn/checkpoint.pth \
  --device cuda:0
  
# 如果运行无错误，说明代码逻辑正确
```

**验证清单**:
- [ ] 模型加载成功
- [ ] 观测维度匹配 (78 = 39 * 2 for ProprioceptionObsCfg)
- [ ] 动作处理正常 (EMA 平滑)
- [ ] 无运行时错误

#### 步骤 4: 部署到真实硬件 ⚠️ **需要 ROS2 环境**

**前置条件**:
1. 安装 LEAP Hand ROS2 SDK: https://github.com/leap-hand/LEAP_Hand_API/tree/main/ros2_module
2. 配置 USB 延迟优化 (参考官方文档)
3. 启动 LEAP Hand ROS2 节点

**部署步骤**:

```bash
# 终端 1: 启动 LEAP Hand ROS2 节点
cd /path/to/LEAP_Hand_API/ros2_module
ros2 launch launch_leap.py

# 终端 2: 运行 Sim2Real 部署脚本
cd /home/hac/isaac/AnyRotate
source /home/hac/isaac/env_isaaclab/bin/activate

python scripts/rl_games/sim2real_play.py \
  --checkpoint logs/rl_games/leaphand_object_rot/2025-11-17_20-43-31/nn/checkpoint.pth \
  --use_ros2 \
  --device cuda:0
```

**安全注意事项**:
- ⚠️ 首次运行时，确保手附近无障碍物
- ⚠️ 准备好急停按钮或 Ctrl+C 中断
- ⚠️ 检查关节限位配置是否正确

#### 步骤 5: 优化与调试

**常见问题**:

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 动作抖动 | EMA 系数不匹配 | 检查 `alpha=1/24` 是否与训练时一致 |
| 关节位置偏移 | 坐标系转换错误 | 验证 `sim_to_real_indices` |
| 策略性能差 | 观测维度/顺序错误 | 打印观测向量，对比仿真 |
| ROS2 通信延迟 | USB 延迟高 | 优化 USB 设置 (参考官方文档) |

**性能优化**:
- 调整控制频率 (`control_hz`)
- 添加传感器噪音鲁棒性训练
- 微调 EMA 系数以适应真实硬件响应

### 5.4 与官方方案的对比

| 特性 | 官方 reorient_z_ros.py | 本方案 (sim2real_play.py) |
|------|----------------------|--------------------------|
| **环境架构** | DirectRLEnv (硬编码) | ManagerBasedRLEnv (模块化) |
| **观测管理** | 手动拼接 `obs_hist_buf` | `PolicyObservation` 自动管理 |
| **动作处理** | 手动 EMA + 缩放 | `EMAJointPositionAction` 复现 |
| **命令管理** | 无 (固定目标) | `ContinuousRotationCommand` 管理 |
| **代码复用** | ❌ 难以扩展到其他任务 | ✅ 可适配任意 ManagerBasedRLEnv |
| **可维护性** | ⚠️ 硬编码逻辑多 | ✅ 模块化、易调试 |

---

## 5.5 旧版实现方案 (已废弃)

<details>
<summary>点击展开旧版代码（供参考，不推荐使用）</summary>

```python
# 以下代码为旧版实现方案，已被上述模块化方案替代
# 仅供理解设计思路，实际使用请参考 sim2real_play.py

class LeapHandHardwareInterface(Node):
    """LEAP Hand 硬件通信接口 - ROS2 版本"""
    
    def __init__(self, cfg: dict):
        super().__init__('leaphand_sim2real')
        self.cfg = cfg
        
        # ROS2 通信设置
        self.pub_cmd = self.create_publisher(JointState, '/cmd_ones', 10)
        self.cli_state = self.create_client(LeapPosVelEff, '/leap_pos_vel_eff')
        
        # 关节配置（从环境配置读取）
        self.joint_names = cfg["joint_names"]  # ["a_0", "a_1", ...]
        self.dof_limits = self._load_joint_limits()
        
        # 坐标转换索引（从仿真环境获取）
        self.sim_to_real_indices = self._get_sim_to_real_mapping()
        
    def poll_observation(self) -> dict:
        """读取真实机器人状态 → 构造观测字典"""
        # 1. ROS2 服务调用获取关节状态
        response = self.cli_state.call(LeapPosVelEff.Request())
        joint_pos_real = torch.tensor(response.position)  # 真实硬件顺序
        
        # 2. 坐标系转换: Real → Sim
        joint_pos_sim = self._real_to_sim_transform(joint_pos_real)
        
        # 3. 归一化（与训练时一致）
        joint_pos_normalized = self._normalize_joint_pos(joint_pos_sim)
        
        # 4. 构造观测字典（模拟 ObservationManager）
        obs = {
            "joint_pos": joint_pos_normalized,
            # 其他观测项...
        }
        return obs
        
    def send_action(self, action: torch.Tensor):
        """发送动作指令到真实机器人"""
        # 1. 动作后处理（模拟 ActionManager）
        # 注意: 这里需要复现 EMAJointPositionToLimitsAction 的逻辑
        target_pos = self._apply_action_transform(action)
        
        # 2. 坐标系转换: Sim → Real
        target_pos_real = self._sim_to_real_transform(target_pos)
        
        # 3. 发布 ROS 消息
        msg = JointState()
        msg.position = target_pos_real.cpu().numpy().tolist()
        self.pub_cmd.publish(msg)
        
    def _apply_action_transform(self, action: torch.Tensor) -> torch.Tensor:
        """复现 EMAJointPositionToLimitsAction 的逻辑"""
        # 参考: isaaclab/envs/mdp/actions/joint_actions.py
        # 1. EMA 平滑: target = alpha * action + (1-alpha) * prev_target
        # 2. 缩放到关节限位: rescale_to_limits=True
        pass
        
    def _real_to_sim_transform(self, pos_real: torch.Tensor) -> torch.Tensor:
        """真实硬件关节顺序 → 仿真关节顺序"""
        # 使用 sim_to_real_indices (从 env.sim_real_indices() 获取)
        return pos_real[self.real_to_sim_indices]
        
    def _normalize_joint_pos(self, pos_sim: torch.Tensor) -> torch.Tensor:
        """关节位置归一化 (与 mdp.joint_pos_limit_normalized 一致)"""
        lower, upper = self.dof_limits
        return (2.0 * pos_sim - upper - lower) / (upper - lower)
```

#### 步骤 4: 实现策略包装器

```python
# 文件: source/leaphand/deployment/policy_wrapper.py

import torch
import yaml
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder

class RLGamesPolicyWrapper:
    """RL-Games 策略加载与推理"""
    
    def __init__(self, cfg_path: str, ckpt_path: str, obs_group_name: str = "policy"):
        self.device = "cuda:0"
        
        # 1. 加载训练配置
        with open(cfg_path, 'r') as f:
            self.train_cfg = yaml.safe_load(f)
        
        # 2. 推断观测维度（从环境配置读取）
        self.obs_dim = self._infer_obs_dim(obs_group_name)
        self.action_dim = 16  # LEAP Hand 关节数
        
        # 3. 构建网络
        self.model = self._build_network()
        
        # 4. 加载权重
        weights = torch_ext.load_checkpoint(ckpt_path)
        self.model.load_state_dict(weights["model"])
        
        # 5. 初始化 RNN 状态（如果需要）
        if self.model.is_rnn():
            self.hidden_states = self.model.get_default_rnn_state()
            
    def forward(self, obs: dict) -> torch.Tensor:
        """策略推理"""
        # 1. 拼接观测向量（模拟 ObservationManager）
        obs_tensor = self._concat_observations(obs)
        
        # 2. 网络前向传播
        with torch.no_grad():
            batch_dict = {
                "is_train": False,
                "obs": obs_tensor.unsqueeze(0),  # [1, obs_dim]
                "prev_actions": torch.zeros(1, self.action_dim).to(self.device),
            }
            if self.model.is_rnn():
                batch_dict["rnn_states"] = self.hidden_states
                
            result = self.model(batch_dict)
            action = result["mus"][0]  # 确定性策略
            
            if self.model.is_rnn():
                self.hidden_states = result["rnn_states"]
                
        return action
        
    def _concat_observations(self, obs: dict) -> torch.Tensor:
        """按照 ObservationManager 的顺序拼接观测"""
        # 需要与 ProprioceptionObsCfg 的定义顺序一致
        # 示例: [joint_pos, goal_pose, last_action]
        components = []
        for term_name in ["joint_pos", "goal_pose", "last_action"]:
            components.append(obs[term_name])
        return torch.cat(components, dim=-1)
```

#### 步骤 5: 主部署脚本

```python
# 文件: scripts/deployment/deploy_leaphand_ros2.py

#!/usr/bin/env python3
import rclpy
from leaphand.deployment.leaphand_sim2real import LeapHandSim2Real

def main():
    rclpy.init()
    
    # 配置路径
    cfg_path = "source/leaphand/leaphand/tasks/manager_based/leaphand/agents/rl_games_ppo_cfg.yaml"
    ckpt_path = "logs/rl_games/leaphand_object_rot/2025-11-17_20-43-31/nn/checkpoint.pth"
    
    hardware_cfg = {
        "joint_names": ["a_0", "a_1", ...],  # 从环境配置读取
        "init_pose": [0.0, 0.5, ...],  # 初始预抓取姿态
        "control_hz": 30,
    }
    
    # 创建部署实例
    deployer = LeapHandSim2Real(cfg_path, ckpt_path, hardware_cfg)
    
    try:
        deployer.deploy()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
```

### 5.3 关键实现细节

#### 细节 1: 动作处理逻辑复现

您的环境使用 `EMAJointPositionToLimitsActionCfg`，其核心逻辑为:

```python
# isaaclab/envs/mdp/actions/joint_actions.py (参考)
class EMAJointPositionToLimitsAction:
    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        # 1. EMA 平滑
        self.prev_target = self.alpha * action + (1 - self.alpha) * self.prev_target
        
        # 2. 缩放到关节限位 (如果 rescale_to_limits=True)
        if self.rescale_to_limits:
            # 假设 action ∈ [-1, 1]
            target = (action + 1) / 2  # → [0, 1]
            target = self.lower_limits + target * (self.upper_limits - self.lower_limits)
        else:
            target = self.prev_target
            
        return torch.clamp(target, self.lower_limits, self.upper_limits)
```

**部署时必须复现此逻辑**，否则动作分布会不匹配。

#### 细节 2: 观测拼接顺序

ManagerBasedRLEnv 的观测拼接顺序由 `ObservationManager` 决定，通常按配置文件中定义的顺序:

```python
# inhand_base_env_cfg.py 中的定义顺序
class ProprioceptionObsCfg(ObsGroup):
    joint_pos = ObsTerm(...)      # 第 1 项: 16 维
    goal_pose = ObsTerm(...)       # 第 2 项: 7 维 (pos + quat)
    last_action = ObsTerm(...)     # 第 3 项: 16 维
    
# 最终拼接: [joint_pos(16), goal_pose(7), last_action(16)] = 39 维
# 历史: 39 * 2 = 78 维
```

**部署时必须保持相同的拼接顺序**。

#### 细节 3: 坐标系转换

LEAP Hand 的仿真关节顺序与真实硬件不同，需要索引映射:

```python
# 从官方 reorientation_env.py 的 sim_real_indices() 获取
sim_to_real_indices = torch.tensor([4, 0, 8, 12, 6, 2, 10, 14, 7, 3, 11, 15, 1, 5, 9, 13])
real_to_sim_indices = torch.tensor([1, 12, 5, 9, 0, 13, 4, 8, 2, 14, 6, 10, 3, 15, 7, 11])

# 使用方法
joint_pos_sim = joint_pos_real[real_to_sim_indices]
joint_cmd_real = joint_cmd_sim[sim_to_real_indices]
```

---

## 6. 总结与建议

### 6.1 核心结论

| 问题 | 答案 |
|-----|------|
| **1. 官方 sim2real 适用范围** | ❌ 仅适用于 DirectRLEnv 的 `reorientation_env.py`，不适用于其他环境 |
| **2. 环境架构差异** | ✅ ManagerBasedRLEnv 更模块化，观测/动作/奖励都是声明式配置 |
| **3. IsaacLab 标准 workflow** | ✅ Teacher-Student 蒸馏 (3 阶段)，通过 ObsGroup 区分特权/部署观测 |
| **4. 自定义接口可行性** | ✅ 完全可行且推荐，参考官方思路但适配 ManagerBasedRLEnv 架构 |

### 6.2 推荐行动方案

#### 短期 (1-2 天)
1. ✅ **验证部署观测**: 确认 `ProprioceptionObsCfg` 仅包含可测量项
2. ✅ **测试仿真推理**: 修改 `play.py` 仅使用 `ProprioceptionObsCfg`，验证策略性能

#### 中期 (3-5 天)
3. ✅ **实现硬件接口**: 参考本文档的 `LeapHandHardwareInterface`
4. ✅ **复现动作逻辑**: 在部署代码中精确复现 `EMAJointPositionToLimitsAction`
5. ✅ **坐标系对齐**: 从官方环境获取 `sim_to_real_indices` 并应用

#### 长期 (可选)
6. ⚠️ **Teacher-Student 蒸馏**: 如果策略性能不足，可尝试先用 `PrivilegedObsCfg` 训练 teacher，再蒸馏到 student

### 6.3 与官方方案的差异

| 特性 | 官方 reorient_z_ros.py | 推荐方案 |
|-----|----------------------|---------|
| **环境架构** | DirectRLEnv (硬编码) | ManagerBasedRLEnv (声明式) |
| **代码复用** | 仅复用 `RLGamesPolicy` | 复用 play.py + 自定义硬件接口 |
| **灵活性** | ❌ 固定观测结构 | ✅ 可扩展到任意 ObsGroup |
| **可维护性** | ❌ 硬编码逻辑多 | ✅ 模块化分离 |
| **符合最佳实践** | ⚠️ 部分符合 | ✅ 完全符合 IsaacLab 理念 |

### 6.4 关键风险提示

| 风险 | 缓解措施 |
|-----|---------|
| **观测维度不匹配** | 严格对齐 `ObservationManager` 的拼接顺序 |
| **动作分布偏移** | 精确复现 `EMAJointPositionToLimitsAction` 逻辑 |
| **坐标系错误** | 使用官方验证过的 `sim_to_real_indices` |
| **传感器噪音低估** | 在仿真中增大噪音，或真机收集数据微调 |

---

## 附录 A: 参考资料

- **IsaacLab Sim2Real 文档**: `/home/hac/isaac/IsaacLab/docs/source/experimental-features/newton-physics-integration/sim-to-real.rst`
- **官方 LEAP Hand Sim2Real**: `/home/hac/isaac/LEAP_Hand_Isaac_Lab/source/LEAP_Isaaclab/LEAP_Isaaclab/deployment_scripts/reorient_z_ros.py`
- **Spot Deployment Blog**: https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/

## 附录 B: 快速检查清单

部署前请确认:
- [ ] 策略使用 `ProprioceptionObsCfg` 训练（或从 teacher 蒸馏）
- [ ] 观测拼接顺序与训练时一致
- [ ] 动作处理逻辑与 `EMAJointPositionToLimitsAction` 一致
- [ ] 坐标系转换索引已从官方环境获取
- [ ] 真实传感器噪音 ≥ 仿真噪音
- [ ] 控制频率与训练时一致 (`decimation * sim.dt`)

---

**文档版本**: v1.0  
**维护者**: AI Agent  
**最后更新**: 2025-11-18
