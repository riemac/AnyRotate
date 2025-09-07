#!/usr/bin/env python3

"""根据LeapHand官方工具分析关节限制问题"""

import numpy as np

def LEAPsim_limits():
    """LeapHand仿真环境中的关节限制"""
    sim_min = np.array([-1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366, -0.349, -0.47, -1.20, -1.34])
    sim_max = np.array([1.047,    2.23,  1.885,  2.042,  1.047,   2.23,  1.885,  2.042,  1.047,   2.23,  1.885,  2.042,  2.094,  2.443, 1.90,  1.88])
    return sim_min, sim_max

def check_joint_limits():
    """检查关节限制问题"""

    # 关节名称
    joint_names = [
        'a_0', 'a_1', 'a_2', 'a_3',    # 食指
        'a_4', 'a_5', 'a_6', 'a_7',    # 中指
        'a_8', 'a_9', 'a_10', 'a_11',  # 无名指
        'a_12', 'a_13', 'a_14', 'a_15' # 拇指
    ]

    # 获取仿真限制
    sim_min, sim_max = LEAPsim_limits()

    print("LeapHand仿真关节限制 (来自官方工具):")
    print("=" * 60)
    for i, name in enumerate(joint_names):
        print(f"{name:>6}: [{sim_min[i]:7.3f}, {sim_max[i]:7.3f}]")

    print("\n检查配置中的初始关节位置:")
    print("=" * 60)

    # 检查配置文件中的初始位置
    init_joint_pos = np.array([
        0.000, 0.500, 0.000, 0.000,    # 食指
        -0.750, 1.300, 0.000, 0.750,   # 中指
        1.750, 1.500, 1.750, 1.750,    # 无名指
        0.000, 1.000, 0.000, 0.000,    # 拇指
    ])

    violations = []
    for i, name in enumerate(joint_names):
        pos = init_joint_pos[i]
        lower, upper = sim_min[i], sim_max[i]
        if pos < lower or pos > upper:
            violations.append((name, pos, lower, upper))
            print(f"{name:>6}: {pos:7.3f} ❌ NOT in [{lower:7.3f}, {upper:7.3f}]")
        else:
            print(f"{name:>6}: {pos:7.3f} ✓  in [{lower:7.3f}, {upper:7.3f}]")

    if violations:
        print(f"\n发现 {len(violations)} 个违反限制的关节:")
        for name, pos, lower, upper in violations:
            print(f"  - '{name}': {pos} not in [{lower:.3f}, {upper:.3f}]")
            # 建议修正值
            corrected = max(lower, min(upper, pos))
            print(f"    建议修正为: {corrected:.3f}")

        print("\n修正后的关节位置:")
        corrected_pos = np.clip(init_joint_pos, sim_min, sim_max)
        print("joint_pos={")
        for i, name in enumerate(joint_names):
            print(f'    "{name}": {corrected_pos[i]:.3f},')
        print("}")
    else:
        print("\n✓ 所有关节位置都在限制范围内")

if __name__ == "__main__":
    check_joint_limits()
