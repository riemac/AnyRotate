# LeapHand连续旋转任务课程学习指南

本指南介绍了如何在LeapHand连续旋转任务中使用Isaac Lab的Curriculum Manager实现灵活的课程学习策略。

## 概述

我们的课程学习系统提供了三个核心功能：

1. **动态奖励权重调整** - 根据训练进度自动调整不同奖励项的权重
2. **自适应域随机化 (ADR)** - 智能地启用和调整环境参数随机化强度
3. **动态旋转轴复杂度** - 从简单的固定轴旋转逐步过渡到任意轴旋转

## 课程学习配置类型

### 1. 无课程学习 (`NoCurriculumCfg`)
```python
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotNoCurriculumEnvCfg

env_cfg = LeaphandContinuousRotNoCurriculumEnvCfg()
```
- 适用场景：基线对比、调试、简单任务
- 特点：所有参数保持固定，无动态调整

### 2. 仅奖励权重课程学习 (`RewardOnlyCurriculumCfg`)
```python
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotRewardOnlyEnvCfg

env_cfg = LeaphandContinuousRotRewardOnlyEnvCfg()
```
- 适用场景：需要平衡不同奖励项的重要性
- 包含的课程学习项：
  - `grasp_stability_weight`: 抓取稳定性权重调整
  - `rotation_velocity_weight`: 旋转速度权重调整
  - `fall_penalty_weight`: 掉落惩罚权重调整

### 3. 仅自适应域随机化 (`ADROnlyCurriculumCfg`)
```python
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotADROnlyEnvCfg

env_cfg = LeaphandContinuousRotADROnlyEnvCfg()
```
- 适用场景：提高策略鲁棒性，增强泛化能力
- 包含的课程学习项：
  - `object_mass_adr`: 物体质量随机化
  - `friction_adr`: 摩擦系数随机化
  - `gravity_adr`: 重力随机化

### 4. 仅旋转轴复杂度 (`RotationAxisOnlyCurriculumCfg`)
```python
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotAxisOnlyEnvCfg

env_cfg = LeaphandContinuousRotAxisOnlyEnvCfg()
```
- 适用场景：逐步增加任务复杂度
- 包含的课程学习项：
  - `progressive_axis_complexity`: X轴 → Y轴 → Z轴 → 任意轴

### 5. 基础课程学习 (`BasicCurriculumCfg`)
```python
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotBasicEnvCfg

env_cfg = LeaphandContinuousRotBasicEnvCfg()
```
- 适用场景：初学者，简单的课程学习需求
- 包含：仅奖励权重调整

### 6. 中级课程学习 (`IntermediateCurriculumCfg`)
```python
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotIntermediateEnvCfg

env_cfg = LeaphandContinuousRotIntermediateEnvCfg()
```
- 适用场景：平衡复杂度和训练效率
- 包含：奖励权重调整 + 简单旋转轴复杂度

### 7. 高级课程学习 (`AdvancedCurriculumCfg`)
```python
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg

env_cfg = LeaphandContinuousRotEnvCfg()  # 默认配置
```
- 适用场景：完整的课程学习体验，最佳训练效果
- 包含：所有课程学习功能

### 8. 自定义课程学习 (`CustomizableCurriculumCfg`)
```python
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotCustomEnvCfg

env_cfg = LeaphandContinuousRotCustomEnvCfg()
# 或者自定义配置
env_cfg.curriculum.enable_reward_curriculum = True
env_cfg.curriculum.enable_adr_curriculum = False
env_cfg.curriculum.enable_rotation_axis_curriculum = True
env_cfg.curriculum.reward_curriculum_mode = "step_based"
env_cfg.curriculum.rotation_axis_curriculum_mode = "performance_based"
```
- 适用场景：需要精确控制课程学习行为
- 特点：可以灵活启用/禁用各种课程学习功能

## 课程学习模式

每种课程学习功能都支持多种模式：

### 1. `"disabled"` - 禁用模式
- 完全禁用该课程学习功能
- 参数保持初始值不变

### 2. `"step_based"` - 基于步数模式
- 根据训练步数自动调整参数
- 预定义的时间表，可预测的行为

### 3. `"performance_based"` - 基于性能模式
- 根据策略性能动态调整参数
- 需要环境实现性能跟踪功能

### 4. `"hybrid"` - 混合模式
- 结合步数和性能两种策略
- 更智能的自适应调整

## 课程学习时间表

### 奖励权重调整时间表
- **0-50万步**: 抓取稳定性高权重(2.0)，旋转速度低权重(10.0)
- **50-100万步**: 抓取稳定性中权重(1.5)，旋转速度中权重(15.0)
- **100万步后**: 抓取稳定性正常权重(1.0)，旋转速度高权重(20.0)

### 域随机化启用时间表
- **0-60万步**: 无域随机化
- **60-120万步**: 物体质量随机化逐步启用
- **80-150万步**: 摩擦系数随机化逐步启用
- **100-180万步**: 重力随机化逐步启用

### 旋转轴复杂度时间表
- **0-40万步**: X轴旋转
- **40-80万步**: Y轴旋转
- **80-120万步**: Z轴旋转
- **120万步后**: 任意轴旋转

## 使用示例

### 基本使用
```python
from omni.isaac.lab.envs import ManagerBasedRLEnv
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg

# 创建环境配置（默认使用高级课程学习）
env_cfg = LeaphandContinuousRotEnvCfg()
env_cfg.scene.num_envs = 1024

# 创建环境
env = ManagerBasedRLEnv(cfg=env_cfg)

# 检查课程学习状态
print(f"课程学习项: {env.curriculum_manager.active_terms}")
```

### 自定义配置
```python
from leaphand.tasks.manager_based.leaphand.mdp.curriculum_configs import CustomizableCurriculumCfg

# 创建自定义课程学习配置
custom_curriculum = CustomizableCurriculumCfg(
    enable_reward_curriculum=True,      # 启用奖励权重调整
    enable_adr_curriculum=False,        # 禁用域随机化
    enable_rotation_axis_curriculum=True,  # 启用旋转轴复杂度
    reward_curriculum_mode="step_based",
    rotation_axis_curriculum_mode="hybrid"
)

# 应用到环境配置
env_cfg = LeaphandContinuousRotEnvCfg()
env_cfg.curriculum = custom_curriculum
```

### 运行演示
```bash
# 运行不同的课程学习配置
python scripts/curriculum_demo.py --env_cfg no_curriculum
python scripts/curriculum_demo.py --env_cfg reward_only
python scripts/curriculum_demo.py --env_cfg advanced
python scripts/curriculum_demo.py --env_cfg custom
```

## 最佳实践

1. **开始简单**: 从基础课程学习开始，逐步增加复杂度
2. **监控性能**: 定期检查课程学习状态和训练进度
3. **调整时间表**: 根据具体任务调整课程学习的时间节点
4. **组合使用**: 不同的课程学习功能可以组合使用以获得最佳效果
5. **性能对比**: 使用无课程学习配置作为基线进行对比

## 扩展开发

如需添加新的课程学习功能，请参考 `source/leaphand/leaphand/tasks/manager_based/leaphand/mdp/curriculums.py` 中的实现模式，并在 `curriculum_configs.py` 中添加相应的配置类。
