#!/usr/bin/env python3

"""调试LeapHand机器人的body名称"""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="调试LeapHand机器人的body名称")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from leaphand.robots.leap import LEAP_HAND_CFG

def main():
    """主函数"""

    # 直接从USD文件中读取信息
    from pxr import Usd, UsdGeom
    from pathlib import Path

    usd_path = Path(__file__).parent.parent / "source/leaphand/assets/leap_hand_v1_right/leap_hand_right.usd"
    print(f"读取USD文件: {usd_path}")

    if not usd_path.exists():
        print(f"USD文件不存在: {usd_path}")
        return

    # 打开USD文件
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print("无法打开USD文件")
        return

    # 获取所有的Xform prims（通常对应body）
    xform_prims = []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xform):
            xform_prims.append(prim.GetPath().name)

    print(f"找到的Xform prims: {xform_prims}")

    # 查找可能的手掌body
    palm_candidates = [name for name in xform_prims if 'palm' in name.lower()]
    print(f"可能的手掌body: {palm_candidates}")

    # 查找可能的指尖body
    fingertip_candidates = [name for name in xform_prims if 'tip' in name.lower() or 'finger' in name.lower()]
    print(f"可能的指尖body: {fingertip_candidates}")

    # 查找根部body
    root_candidates = [name for name in xform_prims if 'base' in name.lower() or 'root' in name.lower()]
    print(f"可能的根部body: {root_candidates}")

    # 打印所有prims以供参考
    print(f"\n所有Xform prims:")
    for name in sorted(xform_prims):
        print(f"  - {name}")

if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
