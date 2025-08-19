# LeapHand项目概述

LeapHand项目旨在开发一个基于IsaacLab框架的强化学习环境，用于训练灵巧手进行物体的在手内旋转（in-hand manipulation）任务。该项目使用NVIDIA Isaac Sim作为物理仿真平台。

## 项目结构和关键组件

项目主要分为以下几个部分：

1. **IsaacLab版本**（主要开发目录）：
   - 环境配置：`source/leaphand/leaphand/tasks/direct/leaphand/`
   - 机器人定义：`source/leaphand/leaphand/robots/leaphand.py`
   - 强化学习训练脚本：`scripts/skrl/`

2. **Isaac Gym版本**（参考实现）：
   - `LEAP_Hand_Sim/leapsim/tasks/leap_hand_rot.py`

3. **场景文件**：
   - `source/leaphand/assets/leaphand_object_scene.usda`

## 核心开发任务

根据项目文档和代码分析，您需要重点关注以下几个文件的开发：

### 1. 机器人定义文件 (`leaphand.py`)

当前文件只是一个基础模板，需要根据实际的LeapHand USD模型进行配置：
- 配置正确的USD模型路径
- 定义正确的关节和驱动器配置
- 设置适当的物理参数
- **重要**：该文件应只定义手部本身的配置，不包含环境中其他物体

### 2. 环境配置文件 (`leaphand_env_cfg.py`)

当前文件是基于CartPole环境的模板，需要完全重写以适应LeapHand任务：
- 设置正确的观测空间和动作空间维度
- 配置LeapHand机器人和物体的参数
- 定义奖励函数的权重参数
- 设置环境重置条件和成功条件
- **重要**：采用场景加载模式，通过`scene_asset_cfg`加载完整场景，然后通过引用获取手部和物体实体

### 3. 环境实现文件 (`leaphand_env.py`)

同样基于CartPole环境，需要重写以实现LeapHand特定功能：
- 实现`_setup_scene()`方法添加场景
- 实现`_apply_action()`方法控制手部关节
- 实现`_get_observations()`方法获取观测状态
- 实现`_get_rewards()`方法计算奖励
- 实现`_get_dones()`方法判断终止条件
- 实现`_reset_idx()`方法重置环境状态

## 代码结构设计原则

### 组件化 vs 场景化配置

在设计LeapHand环境时，有两种不同的配置方式：

1. **组件化配置**（参考Allegro Hand实现）：
   - 手部和物体分别定义在不同的USD文件中
   - 环境配置文件中分别加载手部和物体实体
   - 优点：灵活性高，可以独立调整各个组件属性
   - 缺点：配置复杂，需要维护多个文件

2. **场景化配置**（当前LeapHand采用的方式）：
   - 手部和物体在同一个USD场景文件中定义
   - 环境配置文件通过`scene_asset_cfg`加载完整场景
   - 从场景中获取手部和物体的引用
   - 优点：便于可视化编辑和快速原型设计
   - 缺点：灵活性相对较低，依赖USD文件结构

### 推荐的配置结构

无论采用哪种方式，都应该遵循以下原则：

1. **职责分离**：
   - 机器人定义文件([leaphand.py](file:///home/hac/isaac/leaphand/source/leaphand/leaphand/robots/leaphand.py))：专注定义手部物理属性
   - 环境配置文件([leaphand_env_cfg.py](file:///home/hac/isaac/leaphand/source/leaphand/leaphand/tasks/direct/leaphand/leaphand_env_cfg.py))：定义环境相关配置和实体引用
   - 环境实现文件([leaphand_env.py](file:///home/hac/isaac/leaphand/source/leaphand/leaphand/tasks/direct/leaphand/leaphand_env.py))：实现环境逻辑，包括场景加载、实体引用获取和环境状态更新

2. **场景加载规范**：
   - 当USD文件包含完整场景时，使用`scene_asset_cfg`加载整个场景
   - 通过实体引用从场景中获取手部和物体
   - 避免重复定义已经在场景中存在的实体

3. **配置驱动设计**：
   - 通过配置文件控制环境行为
   - 保持代码逻辑与配置分离

目前我更偏向场景化配置

## 参考实现

可以参考以下两个实现来完成开发：

1. **Allegro Hand实现** (`source/leaphand/leaphand/tasks/direct/allegro_hand/`)：
   - 提供了完整的Allegro灵巧手操作环境实现
   - 包含物体操作和目标姿态跟踪的完整逻辑
   - 实现了复杂的奖励函数
   - 展示了组件化配置的最佳实践

2. **Isaac Gym版本的LeapHand** (`LEAP_Hand_Sim/leapsim/tasks/leap_hand_rot.py`)：
   - 提供了LeapHand特定的实现细节
   - 包含了针对LeapHand的奖励设计和控制逻辑

## 开发建议

1. 首先完善`leaphand.py`中的机器人定义，确保USD模型路径和关节数量正确
2. 重写`leaphand_env_cfg.py`配置文件，参考Allegro Hand的配置方式
3. 实现`leaphand_env.py`中的核心方法，参考Allegro Hand和Isaac Gym版本的实现
4. 设计适当的奖励函数，鼓励手部完成物体旋转任务
5. 调整环境参数以获得稳定的训练效果

这个项目的核心是实现一个能够训练LeapHand灵巧手完成物体在手内旋转的强化学习环境，主要挑战在于正确的环境配置、观测空间设计、动作空间定义和奖励函数设计。

---