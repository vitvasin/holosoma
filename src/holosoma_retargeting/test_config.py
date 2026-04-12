from holosoma_retargeting.config_types.data_conversion import DataConversionConfig
from holosoma_retargeting.config_types.robot import RobotConfig

cfg = DataConversionConfig(
    input_file="test",
    robot="g1",
    robot_config=RobotConfig(robot_type="g1", robot_dof=23)
)
print("JOINT_NAMES:", cfg.JOINT_NAMES)
print("Length:", len(cfg.JOINT_NAMES))
