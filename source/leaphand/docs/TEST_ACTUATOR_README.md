# LeapHand 执行器诊断工具

## 概述

`source/leaphand/leaphand/tasks/manager_based/test_actuator.py` 是一个诊断脚本，用于测试 LeapHand 机器人手指的轨迹追踪性能。通过让单个手指跟踪正弦波轨迹，分析不同 **刚度(stiffness)、阻尼(damping) 和动作频率(decimation)** 参数对追踪误差的影响。

## 核心机制

### 1. set_joint_position_target() 如何工作

```
用户调用 robot.set_joint_position_target(q_des)
  ↓
q_des 被写入到 robot.data.joint_pos_target 缓冲区
  ↓
调用 robot.write_data_to_sim() 触发：
  - ImplicitActuator.compute() 计算 PD 控制律:
      tau = stiffness * (q_des - q) + damping * (dq_des - dq) + tau_ff
  - 计算结果写入 PhysX
  ↓
PhysX 每个物理步长都使用相同的 q_des，直到下次更新
```

**结论：一旦设置 q_des，即使不再调用 set_joint_position_target()，物理引擎也会持续维持该目标位置（通过 PD 控制器）。**

### 2. 动作频率与 Decimation

- **物理步长 (physics dt)**: 0.001 s（固定）
- **环境步长 (env dt)**: `dt = physics_dt × decimation`
- **频率关系**: 每个环境步长内，物理引擎循环 `decimation` 次
- **影响**: 
  - 更高的 decimation → 单个命令持续更长物理时间 → 轨迹离散化程度增加

### 3. 关键参数的影响

| 参数 | 增大后的效果 |
|------|-----------|
| **stiffness** | PD 增益更强 → 追踪误差↓，但力量可能饱和 |
| **damping** | 阻尼更强 → 运动更缓，防止过冲 |
| **decimation** | 命令持续时间更长 → 轨迹精度↓ |

## 使用方法

### 基础运行（使用默认参数）

```bash
cd /home/hac/isaac/AnyRotate
source ~/isaac/env_isaaclab/bin/activate
./IsaacLab/isaaclab.sh -p source/leaphand/leaphand/tasks/manager_based/test_actuator.py
```

### 自定义参数

```bash
# 高频率（2.0 Hz）、大振幅、高刚度测试
./IsaacLab/isaaclab.sh -p source/leaphand/leaphand/tasks/manager_based/test_actuator.py \
  --amplitude 0.4 \
  --frequency 2.0 \
  --stiffness 8.0 \
  --damping 1.0 \
  --decimation 4 \
  --duration 15.0

# 低频率（0.5 Hz）、低刚度诊断
./IsaacLab/isaaclab.sh -p source/leaphand/leaphand/tasks/manager_based/test_actuator.py \
  --amplitude 0.2 \
  --frequency 0.5 \
  --stiffness 2.0 \
  --damping 0.2 \
  --duration 20.0
```

### 可用命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `--amplitude` | float | 0.3 | 正弦轨迹振幅（弧度） |
| `--frequency` | float | 1.0 | 正弦轨迹频率（Hz） |
| `--stiffness` | float | 5.0 | 执行器刚度（N/m） |
| `--damping` | float | 0.5 | 执行器阻尼（N·s/m） |
| `--duration` | float | 10.0 | 仿真运行时长（秒） |
| `--decimation` | int | 4 | 动作频率倍数 |

## 输出

脚本会生成以下内容：

### 1. 控制台输出
- 实时打印每 10% 步数的追踪状态（目标位置、实际位置、误差）
- 最终性能报告（RMS 误差、最大误差、平均误差）

### 2. 图表
保存到 `outputs/actuator_diagnostics/` 目录：
- **上图**：目标轨迹 vs 实际轨迹
- **下图**：追踪误差随时间变化
- **标注信息**：仿真参数（频率、振幅、刚度、阻尼、decimation）和性能指标

## 诊断建议

### 如果追踪误差过大？

1. **增加 stiffness**（如从 5.0 → 8.0）
   - 风险：关节力量可能饱和（effort_limit = 0.5 N·m）

2. **减少 decimation**（如从 4 → 2）
   - 更频繁地更新目标位置，轨迹更光滑

3. **降低 frequency**（如从 2.0 → 1.0）
   - 轨迹变化更慢，易于追踪

4. **检查关节限位**
   - 振幅设置是否超出关节可达范围

### 最佳参数范围（经验值）

- **频率**: 0.5 ~ 2.0 Hz（低频为主）
- **振幅**: 0.2 ~ 0.4 rad（≈ 11 ~ 23°）
- **刚度**: 3.0 ~ 10.0 N/m（与指骨刚性相匹配）
- **阻尼**: 0.2 ~ 1.0 N·s/m（选择在不引起过冲的前提下尽量小）

## 实现细节

### SineTrajectoryLogger

记录和分析追踪性能的数据容器：
- 存储目标位置、实际位置、追踪误差
- 计算 RMS、最大、平均误差指标

### 正弦轨迹生成

$$q_{des}(t) = q_{home} + A \sin(2\pi f t)$$

其中：
- $q_{home}$：关节初始位置
- $A$：振幅（弧度）
- $f$：频率（Hz）
- $t$：时间（秒）

### PD 控制律

$$\tau = K_p (q_{des} - q) + K_d (\dot{q}_{des} - \dot{q}) + \tau_{ff}$$

其中：
- $K_p = \text{stiffness}$
- $K_d = \text{damping}$
- $\tau_{ff} = 0$（前馈力矩）

## 扩展方向

如需进一步诊断，可以修改以下内容：

1. **目标关节**：修改 `find_joints()` 正则表达式（当前为 `"index_0"`）
2. **轨迹类型**：替换正弦波为阶跃、三角波等
3. **参数扫描**：多次运行不同参数组合，自动生成对比图表
4. **实时可视化**：集成 Isaac Sim UI，实时显示追踪曲线

## 常见问题

**Q: 为什么要测试单个关节而不是整只手？**  
A: 单关节诊断更容易隔离参数影响，便于对症下药。掌握单关节后可扩展到全手。

**Q: 刚度/阻尼参数从哪里来？**  
A: 当前默认值来自项目配置，可在 `source/leaphand/leaphand/robots/leap.py` 中修改。

**Q: 误差过大导致无法收敛？**  
A: 检查 effort_limit 是否过低（0.5 N·m），或尝试减小 frequency。

---

**生成日期**：2025-11-18  
**脚本路径**：`source/leaphand/leaphand/tasks/manager_based/test_actuator.py`
