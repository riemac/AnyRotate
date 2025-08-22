import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# 使用本地LEAP Hand模型路径
from pathlib import Path

# 使用单独的手部模型文件，而不是包含物体的完整场景
hand_usd_path = str(Path(__file__).parent.parent.parent.parent.parent / "LEAP_Hand_Sim" / "assets" / "leap_hand" / "robot" / "robot.usd")  # 单独的手部USD文件


# LEAP Hand机器人的配置
LEAPHAND_CONFIG = ArticulationCfg(
    # USD资产配置
    spawn=sim_utils.UsdFileCfg(
        usd_path=hand_usd_path,  # USD文件路径 - 使用单独的手部文件
        activate_contact_sensors=False,  # 是否激活接触传感器
        rigid_props=sim_utils.RigidBodyPropertiesCfg(  # 刚体属性配置
            disable_gravity=True,  # 是否禁用重力
            retain_accelerations=False,  # 是否保留加速度
            enable_gyroscopic_forces=False,  # 是否启用陀螺力
            angular_damping=0.01,  # 角阻尼系数
            max_linear_velocity=1000.0,  # 最大线速度
            max_angular_velocity=64 / math.pi * 180.0,  # 最大角速度(转换为度/秒)
            max_depenetration_velocity=1000.0,  # 最大去穿透速度
            max_contact_impulse=1e32,  # 最大接触冲量
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(  # 关节树根节点属性配置
            enabled_self_collisions=True,  # 是否启用自碰撞检测
            solver_position_iteration_count=8,  # 位置求解器迭代次数
            solver_velocity_iteration_count=0,  # 速度求解器迭代次数
            sleep_threshold=0.005,  # 休眠阈值
            stabilization_threshold=0.0005,  # 稳定化阈值
        ),
    ),

    # 机器人初始状态配置
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.095),  # 初始位置(x,y,z)
        rot=(0.0, 1, 0.0, 0.0),  # 初始旋转四元数(w, x, y, z)
        lin_vel=(0.0, 0.0, 0.0),  # 初始线速度(x,y,z)
        ang_vel=(0.0, 0.0, 0.0),  # 初始角速度(x,y,z)
        joint_pos={".*": 0.0},  # 所有关节初始角度设为0
        joint_vel={".*": 0.0},  # 所有关节初始速度设为0
    ),
    
    # init_state=ArticulationCfg.InitialStateCfg( # 这个即使不配置也会默认设为0
    #     pos=(0.0, 0.0, 0.0),  # 初始位置(x,y,z)
    #     rot=(1, 0, 0, 0),  # 初始旋转四元数(w, x, y, z)
    #     lin_vel=(0.0, 0.0, 0.0),  # 初始线速度(x,y,z)
    #     ang_vel=(0.0, 0.0, 0.0),  # 初始角速度(x,y,z)
    #     joint_pos={".*": 0.0},  # 所有关节初始角度设为0
    #     joint_vel={".*": 0.0},  # 所有关节初始速度设为0
    # ),

    # 关节驱动器配置
    actuators={
        "fingers": ImplicitActuatorCfg(  # 手指关节驱动器
            joint_names_expr=[".*"],  # 匹配所有关节名称的正则表达式
            effort_limit=0.5,  # 最大输出力矩限制
            velocity_limit=100.0,  # 最大速度限制
            stiffness=3.0,  # PD控制器刚度系数
            damping=0.1,  # PD控制器阻尼系数
            friction=0.01,  # 关节摩擦系数
        ),
    },
    soft_joint_pos_limit_factor=1.0,  # 关节位置软限位系数
)
