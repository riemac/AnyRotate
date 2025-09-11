# LeapHand动作增量因子动态调整功能使用指南

## 🎯 功能概述

动作增量因子动态调整功能解决了LeapHand连续旋转任务中固定缩放因子导致的训练问题：

- **前期探索不足**：缩放因子过小，策略探索过于保守
- **后期利用欠佳**：缩放因子仍然过小，精细控制效果不佳

通过动态调整机制，实现：
- **前期大缩放因子**：利于探索，快速学习基本策略
- **后期小缩放因子**：利于精细控制，提高任务完成质量
- **平滑过渡**：线性递减，避免突变影响训练稳定性

## 🔧 技术实现

### 核心机制

使用Isaac Lab的课程学习框架（CurrTerm + modify_term_cfg）：

```python
action_scale_factor = CurrTerm(
    func=mdp.modify_term_cfg,
    params={
        "address": "actions.hand_joint_pos.scale",  # 修改动作配置的scale参数
        "modify_fn": leaphand_mdp.modify_action_scale_factor,
        "modify_params": {
            "alpha_max": 0.15,    # 起始缩放因子
            "alpha_min": 0.05,    # 终止缩放因子  
            "start_step": 0,      # 开始调整的步数
            "end_step": 1920000   # 结束调整的步数
        }
    }
)
```

### 调整函数

提供两种调整接口：

1. **基于步数调整** (`modify_action_scale_factor`)
2. **基于轮次调整** (`modify_action_scale_factor_epochs`)

## 📋 使用方法

### 方法1：使用预设环境配置

```python
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import (
    LeaphandContinuousRotActionScaleEnvCfg
)

# 创建包含动作缩放调整的环境
env_cfg = LeaphandContinuousRotActionScaleEnvCfg()
env = ManagerBasedRLEnv(cfg=env_cfg)
```

### 方法2：选择不同调整策略

```python
# 保守型调整：缓慢递减，适合稳定训练
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import (
    LeaphandContinuousRotConservativeScaleEnvCfg
)
env_cfg = LeaphandContinuousRotConservativeScaleEnvCfg()

# 激进型调整：快速递减，适合快速收敛  
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import (
    LeaphandContinuousRotAggressiveScaleEnvCfg
)
env_cfg = LeaphandContinuousRotAggressiveScaleEnvCfg()
```

### 方法3：自定义调整参数

```python
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import (
    LeaphandContinuousRotEnvCfg
)
from leaphand.tasks.manager_based.leaphand.mdp import curriculums as leaphand_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab.envs.mdp as mdp

# 创建基础环境配置
env_cfg = LeaphandContinuousRotEnvCfg()

# 自定义课程学习配置
@configclass
class CustomCurriculumCfg:
    my_action_scale = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "actions.hand_joint_pos.scale",
            "modify_fn": leaphand_mdp.modify_action_scale_factor,
            "modify_params": {
                "alpha_max": 0.20,    # 自定义起始值
                "alpha_min": 0.03,    # 自定义终止值
                "start_step": 50000,  # 自定义开始步数
                "end_step": 500000    # 自定义结束步数
            }
        }
    )

# 应用自定义配置
env_cfg.curriculum = CustomCurriculumCfg()
```

## 🧪 测试验证

使用提供的测试脚本验证功能：

```bash
# 激活环境
cd ~/isaac && source .venv/bin/activate
cd leaphand

# 测试默认策略
python scripts/test_action_scale_curriculum.py --strategy default --num_envs 4

# 测试保守型策略
python scripts/test_action_scale_curriculum.py --strategy conservative --num_envs 4

# 测试激进型策略  
python scripts/test_action_scale_curriculum.py --strategy aggressive --num_envs 4
```

## 📊 预设调整策略

### 默认策略 (LeaphandContinuousRotActionScaleEnvCfg)
- **起始值**: 0.15 (中等探索)
- **终止值**: 0.05 (精细控制)
- **调整区间**: 0-80轮次 (1,920,000步)
- **适用场景**: 通用训练，平衡探索与利用

### 保守型策略 (LeaphandContinuousRotConservativeScaleEnvCfg)
- **起始值**: 0.12 (温和探索)
- **终止值**: 0.08 (温和控制)
- **调整区间**: 20-150轮次
- **适用场景**: 稳定训练，避免过度探索

### 激进型策略 (LeaphandContinuousRotAggressiveScaleEnvCfg)
- **起始值**: 0.25 (强探索)
- **终止值**: 0.03 (超精细控制)
- **调整区间**: 0-50轮次
- **适用场景**: 快速收敛，适合有经验的超参数

## ⚙️ 参数调优建议

### alpha_max (起始缩放因子)
- **0.10-0.15**: 温和探索，适合稳定训练
- **0.15-0.20**: 中等探索，通用选择
- **0.20-0.30**: 强探索，适合复杂任务

### alpha_min (终止缩放因子)
- **0.08-0.10**: 温和精细控制
- **0.05-0.08**: 中等精细控制，通用选择
- **0.02-0.05**: 超精细控制，适合高精度任务

### 调整区间
- **短期调整** (20-50轮次): 快速收敛，可能不够稳定
- **中期调整** (50-100轮次): 平衡选择
- **长期调整** (100-200轮次): 稳定训练，收敛较慢

## 🔍 监控与调试

### 训练过程监控
```python
# 在训练循环中监控缩放因子变化
action_term = env.action_manager.get_term("hand_joint_pos")
current_scale = float(action_term._scale)
print(f"当前动作缩放因子: {current_scale:.6f}")
```

### 日志记录
课程学习状态会自动记录到训练日志中：
- `Curriculum/action_scale_factor`: 当前缩放因子值

## 🚨 注意事项

1. **兼容性**: 仅适用于RelativeJointPositionActionCfg动作配置
2. **参数范围**: alpha_max必须大于alpha_min
3. **步数设置**: end_step必须大于start_step
4. **训练稳定性**: 避免过于激进的参数设置
5. **环境重置**: 缩放因子调整在环境重置时不会重置，保持全局进度

## 📈 预期效果

正确配置后，应该观察到：
- **训练前期**: 动作变化较大，策略快速探索
- **训练中期**: 动作变化逐渐减小，策略逐步收敛
- **训练后期**: 动作变化很小，策略精细调优
- **整体趋势**: 任务完成率和奖励逐步提升
