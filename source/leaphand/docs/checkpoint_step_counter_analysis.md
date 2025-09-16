# Isaac Lab Checkpoint步数计数器分析报告

## 问题确认

通过对Isaac Lab源码分析和实际测试，我们确认了以下关键事实：

### 核心发现

1. env.common_step_counter在checkpoint恢复时不会继承
2. 课程学习权重基于common_step_counter的步数，恢复时从0开始会导致权重错误

## 解决方案

### 方案A: 训练脚本中手动同步步数计数器

- 优点: 简单直接，不修改任务配置
- 缺点: 需在训练脚本中增加逻辑

示例：
```python
if args_cli.checkpoint is not None:
    expected_steps = 23976000  # 根据实际情况调整
    env.unwrapped.common_step_counter = expected_steps
    if hasattr(env.unwrapped, 'curriculum_manager'):
        env.unwrapped.curriculum_manager.compute()
```

### 方案B: 调整任务配置中的课程学习阈值

- 优点: 一次配置，久效
- 缺点: 需结合checkpoint训练步数重新设定阈值

示例（思想）：
```python
def modify_action_penalty_weight(..., late_step: int = max(0, 20_000_000 - 23_976_000)) -> float:
    pass
```

## 实用工具

- scripts/quick_reward_weight_check.py - 快速检查权重变化
- scripts/utils/checkpoint_step_sync.py - 步数同步工具函数

## 建议

持续学习场景：优先方案A；新任务配置：优先方案B。

## 结论

- 问题确实存在且为框架设计行为
- 两条路径均可稳定解决
- 应根据使用场景选择具体方案

## 相关文件

- 任务配置: source/leaphand/leaphand/tasks/manager_based/leaphand/leaphand_continuous_rot_env_cfg.py
- 课程学习: source/leaphand/leaphand/tasks/manager_based/leaphand/mdp/curriculums.py
