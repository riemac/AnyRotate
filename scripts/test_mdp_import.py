#!/usr/bin/env python3

"""测试MDP模块导入"""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="测试MDP模块导入")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

try:
    print("尝试导入MDP模块...")
    from leaphand.leaphand.tasks.manager_based.leaphand import mdp as leaphand_mdp
    print("MDP模块导入成功!")

    # 检查函数是否存在
    functions_to_check = [
        'rotation_axis',
        'rotation_velocity_reward',
        'grasp_reward',
        'stability_reward',
        'object_falling_termination',
        'fingertip_positions',
        'relative_fingertip_positions',
        'object_pose_w',
        'object_velocity_w'
    ]

    print("\n检查函数是否存在:")
    for func_name in functions_to_check:
        if hasattr(leaphand_mdp, func_name):
            print(f"✓ {func_name}")
        else:
            print(f"✗ {func_name} - 不存在!")

    print(f"\n模块中的所有属性: {dir(leaphand_mdp)}")

except ImportError as e:
    print(f"导入失败: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"其他错误: {e}")
    import traceback
    traceback.print_exc()

# close sim app
simulation_app.close()
