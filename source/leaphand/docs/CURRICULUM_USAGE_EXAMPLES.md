# LeapHand课程学习系统使用示例

## 概述

本文档提供了LeapHand连续旋转任务课程学习系统的详细使用示例，展示了如何使用Isaac Lab官方风格的声明式配置来实现灵活的课程学习策略。

## 🎯 核心设计理念

我们的课程学习系统完全遵循Isaac Lab的设计理念：

1. **声明式配置**: 像其他Manager一样，在配置文件中声明课程学习项
2. **模块化设计**: 每个课程学习功能都是独立的函数，可以灵活组合
3. **参数化控制**: 所有时间节点和权重都可以通过参数自定义

## 📋 可用的课程学习函数

### 奖励权重调整函数
- `modify_grasp_stability_weight`: 抓取稳定性权重调整
- `modify_rotation_velocity_weight`: 旋转速度权重调整  
- `modify_fall_penalty_weight`: 掉落惩罚权重调整

### 自适应域随机化函数
- `object_mass_adr`: 物体质量随机化
- `friction_adr`: 摩擦系数随机化
- `gravity_adr`: 重力随机化

### 旋转轴复杂度函数
- `progressive_rotation_axis`: 渐进式旋转轴复杂度（X→Y→Z→任意）
- `simple_rotation_axis`: 简化旋转轴复杂度（Z→任意）
- `custom_rotation_axis`: 自定义旋转轴时间表

## 🚀 使用示例

### 示例1: 使用预设配置

```python
from isaaclab.envs import ManagerBasedRLEnv

# 1. 无课程学习（默认）
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg
env_cfg = LeaphandContinuousRotEnvCfg()

# 2. 完整课程学习
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotFullCurriculumEnvCfg
env_cfg = LeaphandContinuousRotFullCurriculumEnvCfg()

# 3. 仅奖励权重课程学习
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotRewardOnlyEnvCfg
env_cfg = LeaphandContinuousRotRewardOnlyEnvCfg()

# 4. 仅域随机化课程学习
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotADROnlyEnvCfg
env_cfg = LeaphandContinuousRotADROnlyEnvCfg()

# 5. 仅旋转轴复杂度课程学习
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotAxisOnlyEnvCfg
env_cfg = LeaphandContinuousRotAxisOnlyEnvCfg()

# 6. 简化旋转轴课程学习
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotSimpleAxisEnvCfg
env_cfg = LeaphandContinuousRotSimpleAxisEnvCfg()

# 创建环境
env = ManagerBasedRLEnv(cfg=env_cfg)
```

### 示例2: 自定义课程学习配置

```python
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg
from leaphand.tasks.manager_based.leaphand import mdp as leaphand_mdp
import isaaclab.envs.mdp as mdp

@configclass
class MyCustomCurriculumCfg:
    """我的自定义课程学习配置"""
    
    # 自定义奖励权重调整 - 更激进的权重变化
    grasp_stability_weight = CurrTerm(
        func=leaphand_mdp.modify_grasp_stability_weight,
        params={
            "term_name": "grasp_stability",
            "early_weight": 3.0,      # 更高的初期权重
            "mid_weight": 1.8,
            "late_weight": 0.8,       # 更低的后期权重
            "mid_step": 200_000,      # 更早的切换时间
            "late_step": 600_000
        }
    )
    
    # 自定义旋转轴复杂度 - 跳过Y轴
    custom_rotation_axis = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.rotation_axis.rotation_axis_mode",
            "modify_fn": leaphand_mdp.custom_rotation_axis,
            "modify_params": {
                "axis_schedule": {
                    0: "x_axis",           # 0-50万步：X轴
                    500_000: "z_axis",     # 50-100万步：Z轴
                    1_000_000: "random"    # 100万步后：任意轴
                }
            }
        }
    )
    
    # 延迟启用的域随机化
    delayed_mass_adr = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "scene.object.spawn.mass_props.mass",
            "modify_fn": leaphand_mdp.object_mass_adr,
            "modify_params": {
                "enable_step": 1_000_000,    # 更晚启用
                "max_strength_step": 2_000_000,
                "max_variation": 0.5          # 更大的变化幅度
            }
        }
    )

# 应用自定义配置
env_cfg = LeaphandContinuousRotEnvCfg()
env_cfg.curriculum = MyCustomCurriculumCfg()
```

### 示例3: 动态修改课程学习参数

```python
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotFullCurriculumEnvCfg

# 创建配置并修改参数
env_cfg = LeaphandContinuousRotFullCurriculumEnvCfg()

# 修改抓取稳定性权重的时间节点
env_cfg.curriculum.grasp_stability_weight.params["mid_step"] = 300_000
env_cfg.curriculum.grasp_stability_weight.params["late_step"] = 800_000

# 修改旋转轴复杂度的切换时间
env_cfg.curriculum.progressive_rotation_axis.params["modify_params"]["y_axis_step"] = 300_000
env_cfg.curriculum.progressive_rotation_axis.params["modify_params"]["z_axis_step"] = 600_000
env_cfg.curriculum.progressive_rotation_axis.params["modify_params"]["random_axis_step"] = 1_000_000

# 禁用某个课程学习项（通过删除属性）
delattr(env_cfg.curriculum, 'gravity_adr')
```

### 示例4: 组合不同的课程学习策略

```python
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import (
    LeaphandContinuousRotEnvCfg,
    RewardOnlyCurriculumCfg,
    SimpleRotationAxisCurriculumCfg
)

@configclass
class CombinedCurriculumCfg(RewardOnlyCurriculumCfg, SimpleRotationAxisCurriculumCfg):
    """组合奖励权重调整和简化旋转轴课程学习"""
    pass

# 使用组合配置
env_cfg = LeaphandContinuousRotEnvCfg()
env_cfg.curriculum = CombinedCurriculumCfg()
```

## 🧪 测试和调试

### 检查课程学习状态

```python
# 创建环境后检查课程学习配置
env = ManagerBasedRLEnv(cfg=env_cfg)

if hasattr(env, 'curriculum_manager') and env.curriculum_manager is not None:
    print(f"课程学习项数量: {len(env.curriculum_manager.active_terms)}")
    print(f"活跃的课程学习项: {env.curriculum_manager.active_terms}")
    
    # 获取课程学习状态
    curriculum_state = env.curriculum_manager.get_state()
    print(f"当前课程学习状态: {curriculum_state}")
else:
    print("未配置课程学习")
```

### 运行演示脚本

```bash
# 在Isaac Lab环境中运行
cd /home/hac/isaac && source .venv/bin/activate

# 测试不同的课程学习配置
python /home/hac/isaac/leaphand/scripts/curriculum_demo.py --env_cfg no_curriculum --num_envs 16
python /home/hac/isaac/leaphand/scripts/curriculum_demo.py --env_cfg reward_only --num_envs 16
python /home/hac/isaac/leaphand/scripts/curriculum_demo.py --env_cfg full_curriculum --num_envs 16
```

## 📊 课程学习时间表参考

### 默认时间表

| 功能 | 阶段1 | 阶段2 | 阶段3 | 阶段4 |
|------|-------|-------|-------|-------|
| **抓取稳定性权重** | 0步: 2.0 | 50万步: 1.5 | 100万步: 1.0 | - |
| **旋转速度权重** | 0步: 10.0 | 30万步: 15.0 | 80万步: 20.0 | - |
| **掉落惩罚权重** | 0步: -50.0 | 60万步: -100.0 | 120万步: -150.0 | - |
| **物体质量ADR** | 0-60万步: 关闭 | 60-120万步: 0%→30% | 120万步后: 30% | - |
| **摩擦系数ADR** | 0-80万步: 关闭 | 80-150万步: 0%→50% | 150万步后: 50% | - |
| **重力ADR** | 0-100万步: 关闭 | 100-180万步: 0%→20% | 180万步后: 20% | - |
| **旋转轴复杂度** | 0-40万步: X轴 | 40-80万步: Y轴 | 80-120万步: Z轴 | 120万步后: 任意轴 |

## 🎉 总结

通过这个课程学习系统，您可以：

1. ✅ **灵活配置**: 像配置其他Manager一样配置课程学习
2. ✅ **模块化组合**: 自由选择和组合不同的课程学习策略
3. ✅ **参数化控制**: 精确控制每个课程学习项的时间节点和参数
4. ✅ **易于扩展**: 简单添加新的课程学习函数
5. ✅ **完全兼容**: 与Isaac Lab架构完美集成

系统已准备就绪，开始您的课程学习训练之旅吧！🚀
