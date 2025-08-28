# 非对称Actor-Critic (AAC) 实现总结

## 概述

成功为LeapHand连续旋转任务环境实现了非对称Actor-Critic (AAC)训练模式，遵循Sim-to-Real最佳实践，将数据流严格划分为Actor观测空间和Critic状态空间。

## 核心特性

### 1. 配置驱动的观测架构

在`leaphand_continuous_rot_env_cfg.py`中实现了高度可配置的观测系统：

```python
observations_cfg = {
    "actor": {
        "history_steps": 4,  # Actor历史窗口长度
        "components": {
            "dof_pos": True,        # 手部关节角度 (16维)
            "dof_vel": True,        # 手部关节速度 (16维)
            "fingertip_pos": True,  # 指尖位置 (12维)
            "last_action": True,    # 上一个时间步的动作 (16维)
            "rotation_axis": True,  # 当前任务的目标旋转轴 (3维)
        }
    },
    "critic": {
        "history_steps": 4,  # Critic历史窗口长度
        "components": {
            # 继承Actor的所有组件
            "dof_pos": True,
            "dof_vel": True,
            "fingertip_pos": True,
            "last_action": True,
            "rotation_axis": True,
            # Critic独有的特权信息
            "object_pose": True,        # 物体位姿 (7维)
            "object_vel": True,         # 物体线速度和角速度 (6维)
            "dof_torque": True,         # 手部关节力矩 (16维)
            "object_properties": True,  # 物体物理属性 (1维)
        }
    }
}
```

### 2. Actor观测空间 (真实世界可获取)

**基础维度**: 63维
- `dof_pos`: 手部关节角度 (16维)
- `dof_vel`: 手部关节速度 (16维)  
- `fingertip_pos`: 指尖位置，相对于手部基座的局部坐标 (12维)
- `last_action`: 上一个时间步的动作 (16维)
- `rotation_axis`: 当前任务的目标旋转轴 (3维)

**包含历史**: 252维 (63 × 4步历史)

### 3. Critic状态空间 (包含特权信息)

**基础维度**: 93维
- **Actor所有组件** (63维)
- **特权信息**:
  - `object_pose`: 物体位姿，相对于环境局部坐标系 (7维)
  - `object_vel`: 物体线速度和角速度 (6维)
  - `dof_torque`: 手部关节力矩 (16维)
  - `object_properties`: 物体物理属性如质量 (1维)

**包含历史**: 372维 (93 × 4步历史)

### 4. 历史缓冲区机制

- 使用Isaac Lab的`CircularBuffer`实现滑动窗口
- Actor和Critic分别维护独立的历史缓冲区
- 支持可配置的历史窗口长度
- 自动处理环境重置时的缓冲区清理

### 5. 关键设计原则

#### Sim-to-Real兼容性
- Actor观测空间严格限制为真实世界可通过传感器获取的信息
- 指尖位置使用相对于手部基座的局部坐标，避免全局坐标依赖
- 物体位姿使用相对于环境局部坐标系，防止并行训练时坐标偏移泄露

#### 配置驱动设计
- 所有观测组件可独立启用/禁用
- 历史窗口长度可分别配置
- 易于扩展新的观测组件

## 实现文件

### 修改的文件

1. **`leaphand_continuous_rot_env_cfg.py`**
   - 添加`observations_cfg`配置字典
   - 启用`asymmetric_obs = True`
   - 设置正确的`observation_space`和`state_space`维度

2. **`leaphand_continuous_rot_env.py`**
   - 添加历史缓冲区初始化和管理
   - 实现`_get_component_observation()`方法
   - 实现`_build_actor_observation()`方法
   - 实现`_build_critic_state()`方法
   - 修改`_get_observations()`返回AAC格式
   - 添加`_reset_history_buffers()`方法

### 测试文件

- **`test_aac_isaac_sim.py`**: Isaac Sim环境下的完整功能测试

## 测试结果

✅ **所有测试通过**

- 环境创建成功
- 观测空间维度正确：
  - Actor: (2, 252) ✓
  - Critic: (2, 372) ✓
- 历史缓冲区工作正常
- 多步执行稳定
- 环境重置功能正常

## 使用方法

```python
from source.leaphand.leaphand.tasks.direct.leaphand.leaphand_continuous_rot_env import LeaphandContinuousRotEnv
from source.leaphand.leaphand.tasks.direct.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg

# 创建环境
cfg = LeaphandContinuousRotEnvCfg()
env = LeaphandContinuousRotEnv(cfg)

# 重置环境
obs_dict, extras = env.reset()

# 观测格式
actor_obs = obs_dict["policy"]    # Shape: (num_envs, 252)
critic_state = obs_dict["critic"] # Shape: (num_envs, 372)

# 执行动作
actions = torch.randn(num_envs, 16)
obs_dict, rewards, terminated, truncated, info = env.step(actions)
```

## 代码清理和优化

### 清理内容
- ✅ **移除兼容旧观测方式的代码** - 简化了`_get_observations()`方法
- ✅ **修复历史步数使用** - 正确使用`self.actor_history_steps`和`self.critic_history_steps`
- ✅ **添加历史缓冲区查询功能** - 新增`get_history_info()`方法

### 历史步数说明
- **含义**: 存储**过去N个时间步**的数据（不包括当前时间步）
- **示例**: 如果`history_steps=4`，缓冲区存储t-4, t-3, t-2, t-1四个历史时间步
- **查询**: 使用`env.get_history_info()`可以查看当前缓存状态

## 总结

成功实现了完整的非对称Actor-Critic架构，具备：
- 严格的Sim-to-Real兼容性
- 高度可配置的观测系统
- 稳定的历史缓冲区机制
- 完整的测试验证和代码清理

该实现为LeapHand连续旋转任务提供了强大的AAC训练基础，支持更好的策略学习和现实世界迁移。
