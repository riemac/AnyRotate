import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Jetbot/jetbot.usd" # 从服务器获取的路径
#usd_path = "/home/hac/isaac/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/Jetbot/jetbot.usd" # 本地路径


LEAPHAND_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
    actuators={"wheel_acts": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=None, stiffness=None)},
)