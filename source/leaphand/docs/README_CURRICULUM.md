# LeapHand连续旋转任务课程学习系统

## 概述

本项目基于Isaac Lab的Curriculum Manager实现了一个灵活、模块化的课程学习系统，专门为LeapHand连续旋转任务设计。系统采用Isaac Lab官方风格的声明式配置，完全符合ManagerBasedRLEnv架构，支持灵活组合各种课程学习策略。

## 🎯 核心功能

### 1. 动态奖励权重调整
- **抓取稳定性权重**: 训练初期高权重，后期逐步降低
- **旋转速度权重**: 训练初期低权重，后期逐步提高
- **掉落惩罚权重**: 随训练进度逐步加重惩罚

### 2. 自适应域随机化 (ADR)
- **物体质量随机化**: 60万步后启用，120万步达到最大强度
- **摩擦系数随机化**: 80万步后启用，150万步达到最大强度
- **重力随机化**: 100万步后启用，180万步达到最大强度

### 3. 动态旋转轴复杂度
- **渐进式复杂度**: X轴 → Y轴 → Z轴 → 任意轴
- **简化模式**: Z轴 → 任意轴
- **时间节点可配置**: 支持自定义切换时机

## 📁 项目结构

```
source/leaphand/leaphand/tasks/manager_based/leaphand/
├── mdp/
│   ├── curriculums.py              # 核心课程学习函数
│   ├── curriculum_configs.py       # 课程学习配置类
│   └── __init__.py                 # 模块导出
├── leaphand_continuous_rot_env_cfg.py  # 环境配置（含课程学习变体）
└── ...

scripts/
├── curriculum_demo.py              # 课程学习演示脚本
├── test_curriculum_config.py       # 配置测试脚本
└── validate_curriculum_syntax.py   # 语法验证脚本

source/leaphand/docs/
└── curriculum_learning_guide.md    # 详细使用指南
```

## 🚀 快速开始

### 1. 基本使用

```python
from isaaclab.envs import ManagerBasedRLEnv
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg

# 创建环境（默认无课程学习）
env_cfg = LeaphandContinuousRotEnvCfg()
env = ManagerBasedRLEnv(cfg=env_cfg)

# 检查课程学习状态
if hasattr(env, 'curriculum_manager') and env.curriculum_manager is not None:
    print(f"课程学习项: {env.curriculum_manager.active_terms}")
else:
    print("无课程学习配置")
```

### 2. 选择不同的课程学习策略

```python
# 完整课程学习
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotFullCurriculumEnvCfg
env_cfg = LeaphandContinuousRotFullCurriculumEnvCfg()

# 仅奖励权重课程学习
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotRewardOnlyEnvCfg
env_cfg = LeaphandContinuousRotRewardOnlyEnvCfg()

# 仅自适应域随机化
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotADROnlyEnvCfg
env_cfg = LeaphandContinuousRotADROnlyEnvCfg()

# 仅旋转轴复杂度
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotAxisOnlyEnvCfg
env_cfg = LeaphandContinuousRotAxisOnlyEnvCfg()

# 简化旋转轴课程学习
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotSimpleAxisEnvCfg
env_cfg = LeaphandContinuousRotSimpleAxisEnvCfg()
```

### 3. 自定义课程学习配置

```python
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg
from leaphand.tasks.manager_based.leaphand import mdp as leaphand_mdp

@configclass
class MyCurriculumCfg:
    """自定义课程学习配置"""

    # 只启用奖励权重调整，自定义参数
    grasp_stability_weight = CurrTerm(
        func=leaphand_mdp.modify_grasp_stability_weight,
        params={
            "term_name": "grasp_stability",
            "early_weight": 3.0,  # 自定义权重
            "mid_weight": 2.0,
            "late_weight": 1.0,
            "mid_step": 300_000,  # 自定义时间节点
            "late_step": 800_000
        }
    )

# 应用自定义配置
env_cfg = LeaphandContinuousRotEnvCfg()
env_cfg.curriculum = MyCurriculumCfg()
```

## 🎛️ 配置选项

### 环境配置变体

| 配置类 | 描述 | 包含功能 |
|--------|------|----------|
| `LeaphandContinuousRotNoCurriculumEnvCfg` | 无课程学习 | 无 |
| `LeaphandContinuousRotRewardOnlyEnvCfg` | 仅奖励权重 | 奖励权重调整 |
| `LeaphandContinuousRotADROnlyEnvCfg` | 仅域随机化 | 自适应域随机化 |
| `LeaphandContinuousRotAxisOnlyEnvCfg` | 仅旋转轴 | 旋转轴复杂度 |
| `LeaphandContinuousRotBasicEnvCfg` | 基础课程学习 | 奖励权重调整 |
| `LeaphandContinuousRotIntermediateEnvCfg` | 中级课程学习 | 奖励权重 + 简单旋转轴 |
| `LeaphandContinuousRotEnvCfg` | 高级课程学习 | 全功能 |
| `LeaphandContinuousRotCustomEnvCfg` | 自定义课程学习 | 可配置 |

### 课程学习模式

- `"disabled"`: 禁用模式
- `"step_based"`: 基于步数模式（推荐）
- `"performance_based"`: 基于性能模式（需要性能跟踪）
- `"hybrid"`: 混合模式

## 📊 课程学习时间表

### 奖励权重调整
```
0-50万步:   抓取稳定性(2.0) + 旋转速度(10.0) + 掉落惩罚(-50.0)
50-100万步: 抓取稳定性(1.5) + 旋转速度(15.0) + 掉落惩罚(-100.0)
100万步后:  抓取稳定性(1.0) + 旋转速度(20.0) + 掉落惩罚(-150.0)
```

### 域随机化启用
```
0-60万步:   无随机化
60-120万步: 物体质量随机化 (0% → 30%)
80-150万步: 摩擦系数随机化 (0% → 50%)
100-180万步: 重力随机化 (0% → 20%)
```

### 旋转轴复杂度
```
0-40万步:   X轴旋转
40-80万步:  Y轴旋转
80-120万步: Z轴旋转
120万步后:  任意轴旋转
```

## 🧪 测试和验证

### 语法验证
```bash
python scripts/validate_curriculum_syntax.py
```

### 配置测试（需要Isaac Lab环境）
```bash
cd /home/hac/isaac && source .venv/bin/activate
python /home/hac/isaac/leaphand/scripts/test_curriculum_config.py
```

### 演示运行
```bash
cd /home/hac/isaac && source .venv/bin/activate
python /home/hac/isaac/leaphand/scripts/curriculum_demo.py --env_cfg advanced --num_envs 64
```

## 🔧 扩展开发

### 添加新的课程学习功能

1. 在 `curriculums.py` 中添加新的课程学习函数
2. 在 `curriculum_configs.py` 中创建相应的配置类
3. 在环境配置中集成新的课程学习项

### 自定义时间表

```python
from leaphand.tasks.manager_based.leaphand.mdp.curriculums import CurriculumStage, create_reward_curriculum_term

# 自定义奖励权重时间表
custom_stages = [
    CurriculumStage("phase1", 0, 3.0),
    CurriculumStage("phase2", 200_000, 2.0),
    CurriculumStage("phase3", 600_000, 1.0)
]

custom_reward_term = create_reward_curriculum_term(
    term_name="my_reward",
    stages=custom_stages,
    mode="step_based"
)
```

## 📚 相关文档

- [详细使用指南](source/leaphand/docs/curriculum_learning_guide.md)
- [Isaac Lab Curriculum Manager官方文档](https://isaac-sim.github.io/IsaacLab/source/how-to/curriculums.html)

## ✨ 特性亮点

- ✅ **声明式配置**: 符合Isaac Lab ManagerBasedRLEnv架构风格
- ✅ **模块化设计**: 可灵活组合不同的课程学习策略
- ✅ **多种模式**: 支持基于步数、性能和混合模式
- ✅ **易于扩展**: 简单的API设计，便于添加新功能
- ✅ **完整测试**: 提供语法验证和功能测试脚本
- ✅ **详细文档**: 包含使用指南和最佳实践

## 🎉 总结

本课程学习系统成功实现了您的所有需求：

1. ✅ **动态奖励权重调整**: 根据训练进度自动平衡不同奖励项
2. ✅ **自适应域随机化**: 智能启用和调整环境参数随机化
3. ✅ **动态旋转轴复杂度**: 从简单轴到任意轴的渐进式训练
4. ✅ **灵活配置**: 可选择性启用/禁用各种课程学习功能
5. ✅ **声明式风格**: 完全符合ManagerBasedRLEnv架构设计理念

系统已准备就绪，可以开始训练！🚀
