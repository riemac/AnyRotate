# LeapHand连续旋转环境 - 动作平滑配置指南

## 问题背景

原始环境使用直接的绝对位置控制，导致灵巧手动作抖动严重。通过引入动作平滑机制可以有效解决这个问题。

## 解决方案

提供了三种动作控制方案，在 `ActionsCfg` 类中通过注释切换：

### 方案1：原始绝对位置控制（对比用）
```python
hand_joint_pos = mdp.JointPositionToLimitsActionCfg(
    asset_name="robot",
    joint_names=["a_.*"],
    scale=1.0,
    rescale_to_limits=True,
)
```
- **特点**：直接映射动作到关节位置
- **问题**：会产生抖动
- **用途**：仅用于对比测试

### 方案2：EMA指数移动平均平滑（推荐）
```python
hand_joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
    asset_name="robot",
    joint_names=["a_.*"],
    scale=1.0,
    rescale_to_limits=True,
    alpha=0.1,  # 关键参数
)
```
- **原理**：`new_target = α × current_action + (1-α) × previous_target`
- **优势**：可精确控制平滑程度
- **参数调优**：
  - `alpha=0.05`：超强平滑（类似官方LeapHand）
  - `alpha=0.1`：强平滑（推荐起始值）
  - `alpha=0.2`：中等平滑
  - `alpha>0.5`：轻微平滑

### 方案3：相对位置增量控制
```python
hand_joint_pos = mdp.RelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["a_.*"],
    scale=0.05,  # 关键参数
    use_zero_offset=True,
)
```
- **原理**：`new_position = current_position + action_delta`
- **优势**：天然平滑，无历史依赖
- **参数调优**：
  - `scale=0.01-0.03`：超平滑，学习慢
  - `scale=0.05`：平衡选择（推荐起始值）
  - `scale=0.1`：响应快，可能不够平滑

## 使用步骤

### 1. 测试EMA方案（推荐）
1. 确保当前启用方案2（EMA配置）
2. 运行训练：
   ```bash
   cd ~/isaac && source .venv/bin/activate
   cd leaphand
   python scripts/rl_games/train.py --task=Isaac-Leaphand-ContinuousRot-Manager-v0 --num_envs=4
   ```
3. 在Isaac Sim中观察动作平滑度

### 2. 参数调优
如果动作还有抖动：
- 降低 `alpha` 到 0.05（更强平滑）

如果响应太慢：
- 提高 `alpha` 到 0.15-0.2（减少平滑）

### 3. 尝试相对控制方案
如果EMA效果不理想：
1. 注释掉方案2，启用方案3
2. 从 `scale=0.05` 开始测试
3. 根据效果调整scale参数

## 参数对比表

| 方案 | 关键参数 | 平滑度 | 响应速度 | 学习难度 | 推荐场景 |
|------|----------|--------|----------|----------|----------|
| EMA α=0.05 | alpha=0.05 | 极强 | 慢 | 中等 | 精细操作 |
| EMA α=0.1 | alpha=0.1 | 强 | 中等 | 中等 | 连续旋转（推荐） |
| EMA α=0.2 | alpha=0.2 | 中等 | 快 | 容易 | 快速响应 |
| 相对 s=0.03 | scale=0.03 | 强 | 慢 | 难 | 超精细控制 |
| 相对 s=0.05 | scale=0.05 | 中强 | 中等 | 中等 | 平衡选择 |
| 相对 s=0.1 | scale=0.1 | 中等 | 快 | 容易 | 快速学习 |

## 预期效果

成功配置后，您应该观察到：
- ✅ 灵巧手动作变得平滑，无明显抖动
- ✅ 关节运动连续，无突变
- ✅ 训练过程更稳定
- ✅ 物体操作更精确

## 故障排除

如果仍有问题：
1. 检查配置文件语法是否正确
2. 确认只启用了一种方案（其他方案已注释）
3. 尝试不同的参数值
4. 对比原始方案验证改进效果
