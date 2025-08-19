# LeapHand手内旋转任务实现

## 概述

本文档描述了为LeapHand机器人实现的手内旋转（in-hand manipulation/rotation）任务。该任务的目标是让LeapHand机器人在保持抓取物体的同时，将物体旋转到指定的目标姿态。

## 主要修改

### 1. 环境配置文件 (`leaphand_env_cfg.py`)

**关键修改：**
- 调整观测空间维度为61维，包含物体状态、目标旋转、手指状态等
- 优化episode长度为15秒，适合手内旋转任务
- 设置`object_cfg.spawn=None`，使用USDA文件中预定义的物体
- 添加手内旋转任务特定参数：
  - `rotation_reward_scale = 10.0` - 主要旋转奖励
  - `grasp_reward_scale = 5.0` - 抓取保持奖励
  - `stability_reward_scale = 2.0` - 稳定性奖励
  - `rotation_tolerance = 0.1` - 旋转成功容差
  - `target_rotation_range = π` - 目标旋转范围

### 2. 环境实现文件 (`leaphand_env.py`)

**核心功能实现：**

#### 场景设置
- 使用`robot_cfg`加载包含机器人和物体的完整USDA场景
- 物体通过`spawn=None`复用USDA文件中的定义

#### 观测空间设计 (61维)
```python
obs = [
    object_pos,           # 物体位置 (3)
    object_rot,           # 物体旋转 (4)
    goal_rot,             # 目标旋转 (4)
    hand_dof_pos,         # 手指关节位置 (16)
    hand_dof_vel,         # 手指关节速度 (16)
    fingertip_positions,  # 指尖位置 (12)
    object_relative_pos,  # 物体相对位置 (3)
    object_ang_vel,       # 物体角速度 (3)
]
```

#### 奖励函数设计
实现了专门的`compute_hand_rotation_rewards`函数，包含：

1. **旋转奖励** - 主要奖励，鼓励物体朝目标旋转
   ```python
   rotation_reward = rotation_reward_scale * exp(-5.0 * rot_dist)
   ```

2. **抓取奖励** - 鼓励保持物体在手中
   ```python
   grasp_reward = grasp_reward_scale * exp(-10.0 * object_dist)
   ```

3. **稳定性奖励** - 鼓励平稳旋转
   ```python
   stability_reward = stability_reward_scale * exp(-2.0 * |avg_fingertip_dist - 0.05|)
   ```

4. **动作惩罚** - 鼓励平滑动作
5. **跌落惩罚** - 物体掉落时的惩罚
6. **成功奖励** - 达到目标时的奖励

#### 重置逻辑
- 手指重置为基本抓取姿态（轻微弯曲）
- 物体重置到手中的目标位置
- 生成随机的初始和目标旋转
- 使用轴角表示生成目标旋转

#### 目标生成
- 随机生成旋转轴和角度
- 支持最大π弧度的旋转范围
- 添加轴噪声增加任务多样性

#### 完成条件
- **成功**: 物体旋转到目标姿态（容差内）
- **失败**: 物体掉落（距离目标位置太远）
- **超时**: 达到最大episode长度

## 文件结构

```
source/leaphand/
├── leaphand/
│   ├── tasks/direct/leaphand/
│   │   ├── leaphand_env.py          # 主环境实现
│   │   └── leaphand_env_cfg.py      # 环境配置
│   └── robots/
│       └── leaphand.py              # 机器人配置
├── assets/
│   └── leaphand_object_scene.usda   # 包含机器人和物体的场景文件
├── test_leaphand_env.py             # 测试脚本
└── LEAPHAND_ROTATION_TASK.md        # 本文档
```

## 使用方法

### 1. 测试环境
```bash
cd source/leaphand
python test_leaphand_env.py
```

### 2. 训练示例
```python
from leaphand.tasks.direct.leaphand.leaphand_env import LeaphandEnv
from leaphand.tasks.direct.leaphand.leaphand_env_cfg import LeaphandEnvCfg

# 创建环境
cfg = LeaphandEnvCfg()
env = LeaphandEnv(cfg)

# 重置环境
obs, _ = env.reset()

# 执行动作
actions = torch.randn((env.num_envs, env.action_space))
obs, rewards, terminated, truncated, info = env.step(actions)
```

## 关键特性

1. **专门的手内旋转任务设计** - 奖励函数和重置逻辑专门为旋转任务优化
2. **稳定的抓取机制** - 鼓励保持物体在手中的同时进行旋转
3. **平滑的动作控制** - 通过动作惩罚和移动平均鼓励平滑控制
4. **灵活的目标生成** - 支持任意轴和角度的旋转目标
5. **完整的观测空间** - 包含任务所需的所有关键信息

## 配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rotation_reward_scale` | 10.0 | 旋转奖励权重 |
| `grasp_reward_scale` | 5.0 | 抓取奖励权重 |
| `stability_reward_scale` | 2.0 | 稳定性奖励权重 |
| `rotation_tolerance` | 0.1 | 旋转成功容差(弧度) |
| `target_rotation_range` | π | 最大旋转角度 |
| `fall_dist` | 0.15 | 跌落距离阈值 |
| `episode_length_s` | 15.0 | Episode时长 |

## 注意事项

1. 确保USDA文件包含正确的机器人和物体定义
2. 根据实际机器人性能调整奖励权重
3. 可以通过修改`target_rotation_range`调整任务难度
4. 建议先在少量环境中测试，确认无误后再扩展到大规模训练
