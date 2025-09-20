"""
LEAP hand configs file for IsaacLab.

Modified template from https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/allegro.py
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pathlib import Path

LEAP_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{Path(__file__).parent.parent.parent}/assets/leap_hand_v1_right/leap_hand_right.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0, 
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.5, 0.5, -0.5, 0.5),
        joint_pos={"a_.*": 0.0},
    ),
    actuators={ # 高度柔顺、低刚度的系统，通常由小型电机、腱驱动或欠驱动机构成，本质上不是“刚性”设备
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=0.5, # N·m NOTE: 来自LEAP_Hand_Sim官方官方URDF里的是0.95，但LEAP_Hand_Isaac_Lab官方里是0.5。现实设备暂不知晓
            velocity_limit=100.0, # rad/s
            stiffness=3.0, # N/m
            damping=0.1, # N·s/m
            friction=0.01, # 无单位
            armature=0.001, # kg·m^2，来自LEAP_Hand_Sim官方
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Note: LeapHand各Body名称
# 读取USD文件: /home/hac/isaac/leaphand/source/leaphand/assets/leap_hand_v1_right/leap_hand_right.usd
# LeapHand机器人的主要组成部分:
# - 手掌: palm_lower
# - 食指: mcp_joint -> pip -> dip -> fingertip -> index_tip_head 
# - 拇指: thumb_temp_base -> thumb_pip -> thumb_dip -> thumb_fingertip -> thumb_tip_head
# - 中指: mcp_joint_2 -> pip_2 -> dip_2 -> fingertip_2 -> middle_tip_head
# - 无名指: mcp_joint_3 -> pip_3 -> dip_3 -> fingertip_3 -> ring_tip_head