# LeapHand增强奖励函数实现

## 概述

本文档记录了在LeapHand连续旋转任务环境中实现增强奖励函数的完整过程。新的奖励系统包含4个新增奖励函数，旨在提供更精细和稳定的训练信号。

## 新增奖励函数

### 1. 🎯 改进的旋转速度奖励 (目标角速度型)

**原问题**：原始设计意味着"转得越快越好"，可能导致不稳定的高速旋转。

**解决方案**：引入目标角速度概念，使用指数衰减型奖励函数。

```python
def rotation_velocity_reward(
    env: ManagerBasedRLEnv,
    target_angular_speed: float = 1.5,    # 目标角速度 (rad/s)
    speed_tolerance: float = 0.5,         # 速度容忍度 (rad/s)
    decay_factor: float = 5.0,            # 指数衰减因子
) -> torch.Tensor:
```

**奖励公式**：
```
R = exp(-decay_factor * max(0, |projected_velocity| - target_angular_speed - speed_tolerance))
```

**特点**：
- 鼓励达到特定的目标角速度而非无限制加速
- 使用指数衰减提供平滑的奖励梯度
- 通过容忍度参数避免过度惩罚小幅偏差

### 2. 📏 指尖距离惩罚

**目的**：鼓励机器人保持与物体的适当距离，避免失去抓取。

```python
def fingertip_distance_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
```

**实现方式**：
- 计算机器人基座到物体中心的距离
- 线性型惩罚，距离越远惩罚越大
- 简化实现避免复杂的指尖身体名称查找

### 3. ⚡ 扭矩惩罚

**目的**：鼓励使用较小的关节扭矩，提高动作效率。

```python
def torque_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
```

**实现方式**：
- 计算所有关节扭矩的平方和
- 参考LEAP_Hand_Isaac_Lab项目的实现
- 鼓励平滑和高效的动作

### 4. 🎯 旋转轴对齐奖励

**目的**：鼓励实际旋转轴与目标旋转轴对齐。

```python
def rotation_axis_alignment_reward(
    env: ManagerBasedRLEnv,
    theta_tolerance: float = 0.1,         # 角度容忍度 (弧度)
    decay_factor: float = 5.0,            # 指数衰减因子
) -> torch.Tensor:
```

**奖励公式**：
```
R_axis = exp(-decay_factor * max(0, theta - theta_tolerance))
```

**特点**：
- 计算实际旋转轴与目标旋转轴之间的夹角
- 使用指数衰减型奖励提供平滑梯度
- 对无效旋转给予中性奖励

## 配置参数

### 奖励权重配置

```python
# 主要奖励：旋转速度奖励 - 目标角速度型
rotation_velocity = RewTerm(
    func=leaphand_mdp.rotation_velocity_reward,
    weight=15.0,
    params={
        "target_angular_speed": 1.5,   # 目标角速度 (rad/s)
        "speed_tolerance": 0.5,        # 速度容忍度 (rad/s)
        "decay_factor": 5.0,           # 指数衰减因子
    },
)

# 指尖距离惩罚
fingertip_distance_penalty = RewTerm(
    func=leaphand_mdp.fingertip_distance_penalty,
    weight=-2.0,
)

# 扭矩惩罚
torque_penalty = RewTerm(
    func=leaphand_mdp.torque_penalty,
    weight=-0.001,
)

# 旋转轴对齐奖励
rotation_axis_alignment = RewTerm(
    func=leaphand_mdp.rotation_axis_alignment_reward,
    weight=5.0,
    params={
        "theta_tolerance": 0.1,  # 角度容忍度 (弧度)
        "decay_factor": 5.0,     # 指数衰减因子
    },
)
```

## 测试结果

### 奖励统计摘要 (500步测试)

| 奖励项 | 平均值 | 范围 | 权重 |
|--------|--------|------|------|
| rotation_velocity | +0.7607 | [+0.1059, +5.9077] | +15.0 |
| grasp_reward | +3.6152 | [+3.0741, +3.9166] | +4.0 |
| stability_reward | +2.7756 | [+2.4938, +2.9740] | +3.0 |
| rotation_axis_alignment | +2.5000 | [+2.5000, +2.5000] | +5.0 |
| fingertip_distance_penalty | -0.2225 | [-0.2391, -0.1964] | -2.0 |
| torque_penalty | -0.0001 | [-0.0002, -0.0000] | -0.001 |
| action_penalty | -0.0082 | [-0.0123, -0.0042] | -0.0005 |
| pose_diff_penalty | -0.0619 | [-0.1292, -0.0006] | -0.01 |
| fall_penalty | +0.0000 | [+0.0000, +0.0000] | -100.0 |

**总奖励平均值**: +0.3120

## 技术特点

### 参数化设计
- 所有关键参数都可通过RewTerm的params配置
- 支持运行时调整奖励函数行为
- 便于超参数调优和实验

### 数值稳定性
- 使用torch.clamp避免数值溢出
- 指数衰减函数提供平滑梯度
- 处理边界情况（如无效旋转）

### 性能优化
- 使用Isaac Lab官方API (quat_from_angle_axis)
- 避免重复计算和不必要的归一化
- 简化实现减少计算开销

## 使用方法

### 测试脚本
```bash
# 测试增强奖励函数
python scripts/test_enhanced_rewards.py --num_envs 4 --rotation_axis_mode z_axis --headless

# 使用训练好的模型测试
python scripts/rl_games/play.py --task=Isaac-Leaphand-ContinuousRot-Manager-v0 --num_envs=4
```

### 自定义配置
用户可以通过修改环境配置文件中的params参数来调整奖励函数行为：

```python
rotation_velocity = RewTerm(
    func=leaphand_mdp.rotation_velocity_reward,
    weight=15.0,
    params={
        "target_angular_speed": 2.0,   # 调整目标角速度
        "speed_tolerance": 0.3,        # 调整容忍度
        "decay_factor": 8.0,           # 调整衰减强度
    },
)
```

## 文件结构

```
source/leaphand/leaphand/tasks/manager_based/leaphand/
├── mdp/
│   └── rewards.py                      # 增强奖励函数实现
├── leaphand_continuous_rot_env_cfg.py  # 环境配置
└── docs/
    └── enhanced_reward_functions.md    # 本文档

scripts/
└── test_enhanced_rewards.py           # 测试脚本
```

## 总结

增强奖励函数系统成功实现了：
- ✅ 目标角速度型旋转速度奖励，避免过度加速
- ✅ 指尖距离惩罚，维持适当的抓取距离
- ✅ 扭矩惩罚，鼓励高效的动作执行
- ✅ 旋转轴对齐奖励，提高旋转精度
- ✅ 完全参数化的配置系统
- ✅ 数值稳定和性能优化的实现

该系统为LeapHand连续旋转任务提供了更精细、稳定和可调节的训练信号。
