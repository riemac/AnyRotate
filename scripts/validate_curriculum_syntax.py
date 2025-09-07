#!/usr/bin/env python3

"""
验证课程学习配置文件语法的脚本

该脚本检查课程学习相关文件的语法是否正确，无需导入Isaac Lab。

使用方法:
    python scripts/validate_curriculum_syntax.py

作者: AI Assistant
日期: 2025-01-05
"""

import ast
import os
import sys

def validate_python_syntax(file_path):
    """验证Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析AST
        ast.parse(content)
        return True, None
        
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"其他错误: {e}"


def main():
    """主函数"""
    print("="*60)
    print("LeapHand课程学习配置语法验证")
    print("="*60)
    
    # 要检查的文件列表
    files_to_check = [
        "source/leaphand/leaphand/tasks/manager_based/leaphand/mdp/curriculums.py",
        "source/leaphand/leaphand/tasks/manager_based/leaphand/leaphand_continuous_rot_env_cfg.py"
    ]
    
    passed = 0
    total = len(files_to_check)
    
    for file_path in files_to_check:
        print(f"\n检查文件: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"✗ 文件不存在")
            continue
        
        is_valid, error_msg = validate_python_syntax(file_path)
        
        if is_valid:
            print(f"✓ 语法正确")
            passed += 1
        else:
            print(f"✗ {error_msg}")
    
    print("\n" + "="*60)
    print(f"验证结果: {passed}/{total} 文件语法正确")
    
    if passed == total:
        print("✓ 所有文件语法正确！")
        
        # 额外检查：确保关键类和函数定义存在
        print("\n检查关键定义...")
        
        # 检查curriculums.py中的关键定义
        curriculums_file = "source/leaphand/leaphand/tasks/manager_based/leaphand/mdp/curriculums.py"
        if os.path.exists(curriculums_file):
            with open(curriculums_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            key_definitions = [
                "def modify_grasp_stability_weight",
                "def modify_rotation_velocity_weight",
                "def modify_fall_penalty_weight",
                "def object_mass_adr",
                "def friction_adr",
                "def object_scale_adr",
                "def progressive_rotation_axis",
                "def simple_rotation_axis",
                "def custom_rotation_axis"
            ]
            
            for definition in key_definitions:
                if definition in content:
                    print(f"✓ 找到定义: {definition}")
                else:
                    print(f"✗ 缺少定义: {definition}")
        
        print("✓ 课程学习函数定义检查完成")
        
        # 检查环境配置文件中的关键定义
        env_cfg_file = "source/leaphand/leaphand/tasks/manager_based/leaphand/leaphand_continuous_rot_env_cfg.py"
        if os.path.exists(env_cfg_file):
            with open(env_cfg_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            key_env_configs = [
                "class LeaphandContinuousRotEnvCfg",
                "class CurriculumCfg",
                "class RewardOnlyCurriculumCfg",
                "class ADROnlyCurriculumCfg",
                "class RotationAxisOnlyCurriculumCfg",
                "class SimpleRotationAxisCurriculumCfg",
                "class LeaphandContinuousRotFullCurriculumEnvCfg",
                "class LeaphandContinuousRotRewardOnlyEnvCfg",
                "class LeaphandContinuousRotADROnlyEnvCfg",
                "class LeaphandContinuousRotAxisOnlyEnvCfg",
                "class LeaphandContinuousRotSimpleAxisEnvCfg"
            ]
            
            for env_config in key_env_configs:
                if env_config in content:
                    print(f"✓ 找到环境配置: {env_config}")
                else:
                    print(f"✗ 缺少环境配置: {env_config}")
        
        return 0
    else:
        print("✗ 部分文件存在语法错误，请修复。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
