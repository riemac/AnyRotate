# LeapHand双旋转轴可视化实现

## 概述

本文档记录了在LeapHand连续旋转任务环境中实现双旋转轴可视化功能的完整过程。该功能可以同时显示目标旋转轴和实际旋转轴，帮助用户更好地理解和调试旋转任务。

## 功能特性

### 🎯 双轴可视化
- **目标旋转轴**：红色箭头，显示Command管理器指定的目标旋转方向
- **实际旋转轴**：蓝色箭头，显示物体实际旋转的瞬时轴

### 📍 可视化细节
- **位置**：箭头位于物体上方，蓝色箭头比红色箭头稍高避免重叠
- **方向**：遵循右手螺旋定则，拇指指向箭头方向为正旋转方向
- **更新频率**：
  - 目标轴：仅在command重采样时更新（相对稳定）
  - 实际轴：每个策略步更新（动态变化）

## 架构设计

### 分离式架构
采用职责分离的设计原则：

1. **目标旋转轴可视化**
   - 位置：`rotation_axis_command.py`
   - 职责：Command管理器负责目标命令的可视化
   - 颜色：红色箭头
   - 路径：`/Visuals/Command/target_rotation_axis`

2. **实际旋转轴可视化**
   - 位置：`rotation_velocity_reward`函数中
   - 职责：Reward计算时同步计算实际旋转轴
   - 颜色：蓝色箭头
   - 路径：`/Visuals/Reward/actual_rotation_axis`

### 优势
- **职责清晰**：各自管理自己的可视化逻辑
- **数据解耦**：避免跨模块的数据依赖
- **独立更新**：不同的更新频率和触发条件

## 实现细节

### 核心文件修改

#### 1. `rewards.py` - 实际旋转轴可视化
```python
def rotation_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    visualize_actual_axis: bool = True,
) -> torch.Tensor:
    # ... 计算实际旋转轴 ...
    if visualize_actual_axis:
        _visualize_actual_rotation_axis(env, asset, axis, valid_rotation)
```

#### 2. `commands_cfg.py` - 目标轴可视化配置
```python
def __post_init__(self):
    # 使用Isaac Lab提供的标准红色箭头配置（目标旋转轴）
    self.marker_cfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/target_rotation_axis"
    )
```

#### 3. `leaphand_continuous_rot_env_cfg.py` - 环境配置
```python
rotation_velocity = RewTerm(
    func=leaphand_mdp.rotation_velocity_reward,
    weight=15.0,
    params={
        "asset_cfg": SceneEntityCfg("object"),
        "visualize_actual_axis": True,  # 启用实际旋转轴可视化
    },
)
```

### 技术优化

#### 使用Isaac Lab官方API
- 替换自定义的`_quat_from_two_vectors`函数
- 使用官方的`quat_from_angle_axis`函数
- 减少代码重复，提高可维护性

#### 避免重复计算
- 实际旋转轴在`rotation_velocity_reward`中已被归一化
- 可视化函数直接使用归一化后的轴向量
- 消除不必要的重复归一化操作

## 使用方法

### 测试脚本
```bash
# 运行双旋转轴可视化测试
python scripts/test_dual_rotation_axis_visualization.py --num_envs 4 --rotation_axis_mode random

# 使用训练好的模型测试
python scripts/rl_games/play.py --task=Isaac-Leaphand-ContinuousRot-Manager-v0 --num_envs=4 --checkpoint=path/to/checkpoint.pth
```

### 配置选项
- `visualize_actual_axis`: 控制是否显示实际旋转轴（默认True）
- `debug_vis`: 控制是否显示目标旋转轴（在commands配置中）

## 观察要点

### 正常行为
- **红色箭头**：相对稳定，指向目标旋转轴方向
- **蓝色箭头**：动态变化，反映物体实际旋转状态
- **位置跟随**：两种箭头都跟随物体位置移动

### 调试信息
- 蓝色箭头仅在物体有有效旋转时显示（`valid_rotation > 1e-6`）
- 箭头方向遵循右手螺旋定则
- 多环境场景中每个环境都有独立的箭头显示

## 文件结构

```
source/leaphand/leaphand/tasks/manager_based/leaphand/
├── mdp/
│   ├── commands/
│   │   ├── rotation_axis_command.py    # 目标轴可视化
│   │   └── commands_cfg.py             # 可视化配置
│   └── rewards.py                      # 实际轴可视化
├── leaphand_continuous_rot_env_cfg.py  # 环境配置
└── docs/
    └── dual_rotation_axis_visualization.md  # 本文档

scripts/
└── test_dual_rotation_axis_visualization.py  # 测试脚本
```

## 总结

双旋转轴可视化功能成功实现了：
- ✅ 目标旋转轴和实际旋转轴的同时显示
- ✅ 清晰的颜色区分（红色vs蓝色）
- ✅ 合理的架构设计和职责分离
- ✅ 使用Isaac Lab官方API的优化实现
- ✅ 完整的测试验证

该功能为LeapHand连续旋转任务的开发和调试提供了强有力的可视化支持。
