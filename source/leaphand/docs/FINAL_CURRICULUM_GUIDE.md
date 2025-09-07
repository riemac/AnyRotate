# LeapHand课程学习系统 - 最终使用指南

## 🎉 系统改进完成

经过全面的改进，LeapHand连续旋转任务的课程学习系统现已完全符合Isaac Lab官方架构，并解决了所有技术问题。

## ✅ 已解决的问题

### 1. **EventCfg与ADR课程学习关系** ✅
- **问题**: `mass_distribution_params": (1.0, 1.0)`无随机化效果
- **解决**: ADR函数正确修改EventCfg参数，实现渐进式随机化
- **效果**: 60万步后物体质量从无随机化逐步增强到±50%变化

### 2. **环境坐标系处理** ✅
- **问题**: `fall_penalty`未考虑多环境实例的坐标偏置
- **解决**: 使用`env.scene.env_origins`转换为环境局部坐标系
- **效果**: 多环境训练时掉落检测准确无误

### 3. **自然姿态配置** ✅
- **问题**: `pose_diff_penalty`使用错误的关节名称映射
- **解决**: 采用LEAP_Hand_Isaac_Lab官方的自然姿态配置
- **效果**: 鼓励策略学习更自然的人手姿态

### 4. **物体尺寸域随机化** ✅
- **问题**: 缺少物体尺寸随机化功能
- **解决**: 添加`object_scale_adr`函数和相应的事件配置
- **效果**: 100万步后启用物体尺寸±20%随机化

## 🏗️ 完整的课程学习架构

### 核心组件
```
mdp/
├── curriculums.py          # 9个课程学习MDP函数
├── rewards.py              # 包含pose_diff_penalty和fall_penalty
└── __init__.py             # 模块导出

leaphand_continuous_rot_env_cfg.py  # 环境配置和课程学习变体
```

### 课程学习函数列表
1. **奖励权重调整**:
   - `modify_grasp_stability_weight` - 抓取稳定性权重
   - `modify_rotation_velocity_weight` - 旋转速度权重
   - `modify_fall_penalty_weight` - 掉落惩罚权重

2. **自适应域随机化**:
   - `object_mass_adr` - 物体质量随机化
   - `friction_adr` - 摩擦系数随机化
   - `object_scale_adr` - 物体尺寸随机化

3. **旋转轴复杂度**:
   - `progressive_rotation_axis` - 渐进式复杂度
   - `simple_rotation_axis` - 简化复杂度
   - `custom_rotation_axis` - 自定义时间表

## 📊 完整的课程学习时间表

### 奖励权重调整
```
0-30万步:   抓取稳定性(2.0) + 旋转速度(10.0) + 掉落惩罚(-50.0) + 姿态偏差(-0.01)
30-50万步:  抓取稳定性(2.0) + 旋转速度(15.0) + 掉落惩罚(-50.0) + 姿态偏差(-0.01)
50-80万步:  抓取稳定性(1.5) + 旋转速度(15.0) + 掉落惩罚(-50.0) + 姿态偏差(-0.01)
80-100万步: 抓取稳定性(1.5) + 旋转速度(20.0) + 掉落惩罚(-100.0) + 姿态偏差(-0.02)
100万步后:  抓取稳定性(1.0) + 旋转速度(20.0) + 掉落惩罚(-150.0) + 姿态偏差(-0.02)
```

### 域随机化启用
```
0-60万步:    无域随机化
60-120万步:  物体质量 (1.0,1.0) → (0.5,1.5)
80-150万步:  摩擦系数 (1.0,1.0) → (0.7,1.3)
100-180万步: 物体尺寸 (1.0,1.0) → (0.8,1.2)
```

### 旋转轴复杂度
```
0-40万步:   X轴旋转
40-80万步:  Y轴旋转
80-120万步: Z轴旋转
120万步后:  任意轴旋转
```

## 🚀 使用示例

### 1. 完整课程学习（推荐）
```python
from isaaclab.envs import ManagerBasedRLEnv
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotFullCurriculumEnvCfg

# 创建完整课程学习环境
env_cfg = LeaphandContinuousRotFullCurriculumEnvCfg()
env_cfg.scene.num_envs = 1024

env = ManagerBasedRLEnv(cfg=env_cfg)
print(f"课程学习项: {env.curriculum_manager.active_terms}")
```

### 2. 仅域随机化课程学习
```python
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotADROnlyEnvCfg

env_cfg = LeaphandContinuousRotADROnlyEnvCfg()
env = ManagerBasedRLEnv(cfg=env_cfg)
```

### 3. 自定义课程学习
```python
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotEnvCfg
from leaphand.tasks.manager_based.leaphand import mdp as leaphand_mdp
import isaaclab.envs.mdp as mdp

@configclass
class MyCustomCurriculumCfg:
    """自定义课程学习 - 更激进的ADR"""
    
    # 更早启用物体质量随机化
    object_mass_adr = CurrTerm(
        func=mdp.modify_env_param,
        params={
            "address": "events.object_scale_mass.params.mass_distribution_params",
            "modify_fn": leaphand_mdp.object_mass_adr,
            "modify_params": {
                "enable_step": 300_000,      # 30万步就启用
                "max_strength_step": 800_000, # 80万步达到最大强度
                "max_variation": 0.7          # 更大的变化范围±70%
            }
        }
    )
    
    # 自定义旋转轴时间表
    custom_rotation_axis = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "commands.rotation_axis.rotation_axis_mode",
            "modify_fn": leaphand_mdp.custom_rotation_axis,
            "modify_params": {
                "axis_schedule": {
                    0: "z_axis",           # 从Z轴开始
                    500_000: "x_axis",     # 50万步切换到X轴
                    1_000_000: "random"    # 100万步切换到任意轴
                }
            }
        }
    )

# 应用自定义配置
env_cfg = LeaphandContinuousRotEnvCfg()
env_cfg.curriculum = MyCustomCurriculumCfg()
```

## 🧪 验证和测试

### 语法验证
```bash
cd /home/hac/isaac/leaphand
python scripts/validate_curriculum_syntax.py
```

### 演示运行（需要Isaac Lab环境）
```bash
cd /home/hac/isaac && source .venv/bin/activate
python /home/hac/isaac/leaphand/scripts/curriculum_demo.py --env_cfg full_curriculum --num_envs 64
```

## 📈 预期训练效果

### 训练稳定性提升
- **初期**: 高抓取稳定性权重确保物体不掉落
- **中期**: 逐步提高旋转奖励权重，引导学习旋转技能
- **后期**: 严格的掉落惩罚和姿态约束，确保策略质量

### 泛化能力增强
- **物体质量变化**: ±50%质量变化提高对不同重量物体的适应性
- **摩擦系数变化**: ±30%摩擦变化提高对不同材质的适应性
- **物体尺寸变化**: ±20%尺寸变化提高对不同大小物体的适应性

### 人体工程学改善
- **自然姿态约束**: 鼓励学习更接近人手的自然抓取姿态
- **安全性提升**: 避免过度弯曲或不自然的关节配置
- **可解释性增强**: 策略行为更符合人类直觉

## 🎯 总结

LeapHand课程学习系统现已完全就绪，提供了：

✅ **正确的ADR机制** - 修复了EventCfg参数映射问题
✅ **准确的坐标系处理** - 解决了多环境实例的坐标偏置
✅ **官方的自然姿态** - 使用LEAP_Hand_Isaac_Lab的标准配置
✅ **完整的域随机化** - 质量、摩擦、尺寸三重随机化
✅ **灵活的配置系统** - 6种预设配置 + 自定义配置
✅ **Isaac Lab官方风格** - 完全符合ManagerBasedRLEnv架构

系统已准备好进行高质量的手部操作策略训练！🚀
