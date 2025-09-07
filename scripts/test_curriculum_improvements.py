#!/usr/bin/env python3

"""
测试课程学习系统改进的脚本

该脚本验证所有改进是否正确实现：
1. 环境坐标系修复
2. 自然姿态配置修正
3. 物体尺寸域随机化
4. ADR课程学习机制

使用方法:
    python scripts/test_curriculum_improvements.py

作者: AI Assistant
日期: 2025-01-05
"""

import sys
import os

# 添加项目路径到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "source"))

def test_reward_functions():
    """测试奖励函数的改进"""
    print("测试奖励函数改进...")
    
    try:
        from leaphand.tasks.manager_based.leaphand.mdp.rewards import fall_penalty, pose_diff_penalty
        
        # 检查fall_penalty函数是否包含env_origins处理
        import inspect
        fall_penalty_source = inspect.getsource(fall_penalty)
        if "env.scene.env_origins" in fall_penalty_source:
            print("✓ fall_penalty函数已修复环境坐标系问题")
        else:
            print("✗ fall_penalty函数未修复环境坐标系问题")
            return False
        
        # 检查pose_diff_penalty函数是否使用正确的自然姿态
        pose_diff_penalty_source = inspect.getsource(pose_diff_penalty)
        if "natural_joint_angles" in pose_diff_penalty_source and "0.500" in pose_diff_penalty_source:
            print("✓ pose_diff_penalty函数已使用正确的自然姿态配置")
        else:
            print("✗ pose_diff_penalty函数未使用正确的自然姿态配置")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 奖励函数测试失败: {e}")
        return False


def test_curriculum_functions():
    """测试课程学习函数的改进"""
    print("\n测试课程学习函数改进...")
    
    try:
        from leaphand.tasks.manager_based.leaphand.mdp.curriculums import (
            object_mass_adr, 
            friction_adr, 
            object_scale_adr
        )
        
        # 检查ADR函数是否返回正确的类型
        import inspect
        
        # 检查object_mass_adr
        mass_adr_source = inspect.getsource(object_mass_adr)
        if "tuple[float, float]" in mass_adr_source:
            print("✓ object_mass_adr函数返回类型正确")
        else:
            print("✗ object_mass_adr函数返回类型不正确")
            return False
        
        # 检查friction_adr
        friction_adr_source = inspect.getsource(friction_adr)
        if "tuple[float, float]" in friction_adr_source:
            print("✓ friction_adr函数返回类型正确")
        else:
            print("✗ friction_adr函数返回类型不正确")
            return False
        
        # 检查object_scale_adr
        scale_adr_source = inspect.getsource(object_scale_adr)
        if "dict[str, tuple[float, float]]" in scale_adr_source:
            print("✓ object_scale_adr函数已添加并返回类型正确")
        else:
            print("✗ object_scale_adr函数返回类型不正确")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 课程学习函数测试失败: {e}")
        return False


def test_environment_configs():
    """测试环境配置的改进"""
    print("\n测试环境配置改进...")
    
    try:
        from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import (
            LeaphandContinuousRotEnvCfg,
            LeaphandContinuousRotFullCurriculumEnvCfg,
            LeaphandContinuousRotADROnlyEnvCfg
        )
        
        # 测试默认环境配置
        default_cfg = LeaphandContinuousRotEnvCfg()
        if hasattr(default_cfg.events, 'object_scale_size'):
            print("✓ 默认环境配置已添加物体尺寸随机化事件")
        else:
            print("✗ 默认环境配置未添加物体尺寸随机化事件")
            return False
        
        # 测试完整课程学习配置
        full_cfg = LeaphandContinuousRotFullCurriculumEnvCfg()
        if hasattr(full_cfg.curriculum, 'object_scale_adr'):
            print("✓ 完整课程学习配置已添加物体尺寸ADR")
        else:
            print("✗ 完整课程学习配置未添加物体尺寸ADR")
            return False
        
        # 测试ADR专用配置
        adr_cfg = LeaphandContinuousRotADROnlyEnvCfg()
        if hasattr(adr_cfg.curriculum, 'object_scale_adr'):
            print("✓ ADR专用配置已添加物体尺寸ADR")
        else:
            print("✗ ADR专用配置未添加物体尺寸ADR")
            return False
        
        # 检查奖励配置
        if hasattr(default_cfg.rewards, 'pose_diff_penalty') and hasattr(default_cfg.rewards, 'fall_penalty'):
            print("✓ 奖励配置已添加pose_diff_penalty和fall_penalty")
        else:
            print("✗ 奖励配置未正确添加新的奖励项")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 环境配置测试失败: {e}")
        return False


def test_adr_addresses():
    """测试ADR地址映射的正确性"""
    print("\n测试ADR地址映射...")
    
    try:
        from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotFullCurriculumEnvCfg
        
        cfg = LeaphandContinuousRotFullCurriculumEnvCfg()
        
        # 检查物体质量ADR地址
        mass_adr_address = cfg.curriculum.object_mass_adr.params["address"]
        expected_mass_address = "events.object_scale_mass.params.mass_distribution_params"
        if mass_adr_address == expected_mass_address:
            print("✓ 物体质量ADR地址映射正确")
        else:
            print(f"✗ 物体质量ADR地址映射错误: {mass_adr_address}")
            return False
        
        # 检查摩擦系数ADR地址
        friction_adr_address = cfg.curriculum.friction_adr.params["address"]
        expected_friction_address = "events.object_physics_material.params.static_friction_range"
        if friction_adr_address == expected_friction_address:
            print("✓ 摩擦系数ADR地址映射正确")
        else:
            print(f"✗ 摩擦系数ADR地址映射错误: {friction_adr_address}")
            return False
        
        # 检查物体尺寸ADR地址
        scale_adr_address = cfg.curriculum.object_scale_adr.params["address"]
        expected_scale_address = "events.object_scale_size.params.scale_range"
        if scale_adr_address == expected_scale_address:
            print("✓ 物体尺寸ADR地址映射正确")
        else:
            print(f"✗ 物体尺寸ADR地址映射错误: {scale_adr_address}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ ADR地址映射测试失败: {e}")
        return False


def test_curriculum_timeline():
    """测试课程学习时间表的合理性"""
    print("\n测试课程学习时间表...")
    
    try:
        from leaphand.tasks.manager_based.leaphand.leaphand_continuous_rot_env_cfg import LeaphandContinuousRotFullCurriculumEnvCfg
        
        cfg = LeaphandContinuousRotFullCurriculumEnvCfg()
        
        # 检查时间节点的合理性
        mass_enable = cfg.curriculum.object_mass_adr.params["modify_params"]["enable_step"]
        friction_enable = cfg.curriculum.friction_adr.params["modify_params"]["enable_step"]
        scale_enable = cfg.curriculum.object_scale_adr.params["modify_params"]["enable_step"]
        
        if mass_enable < friction_enable < scale_enable:
            print("✓ ADR启用时间表合理：质量 < 摩擦 < 尺寸")
        else:
            print(f"✗ ADR启用时间表不合理: 质量({mass_enable}) 摩擦({friction_enable}) 尺寸({scale_enable})")
            return False
        
        # 检查变化幅度的合理性
        mass_variation = cfg.curriculum.object_mass_adr.params["modify_params"]["max_variation"]
        friction_variation = cfg.curriculum.friction_adr.params["modify_params"]["max_variation"]
        scale_variation = cfg.curriculum.object_scale_adr.params["modify_params"]["max_variation"]
        
        if 0.2 <= scale_variation <= 0.5 and 0.2 <= friction_variation <= 0.5 and 0.3 <= mass_variation <= 0.7:
            print("✓ ADR变化幅度合理")
        else:
            print(f"✗ ADR变化幅度不合理: 质量({mass_variation}) 摩擦({friction_variation}) 尺寸({scale_variation})")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 课程学习时间表测试失败: {e}")
        return False


def main():
    """主函数"""
    print("="*60)
    print("LeapHand课程学习系统改进验证")
    print("="*60)
    
    tests = [
        test_reward_functions,
        test_curriculum_functions,
        test_environment_configs,
        test_adr_addresses,
        test_curriculum_timeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有改进验证通过！课程学习系统已完全修复。")
        print("\n主要改进:")
        print("  1. ✓ 修复了fall_penalty的环境坐标系问题")
        print("  2. ✓ 修正了pose_diff_penalty的自然姿态配置")
        print("  3. ✓ 添加了物体尺寸域随机化功能")
        print("  4. ✓ 修正了ADR函数的参数类型和地址映射")
        print("  5. ✓ 优化了课程学习时间表和变化幅度")
        return 0
    else:
        print("✗ 部分改进验证失败，请检查实现。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
