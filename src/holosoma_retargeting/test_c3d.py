from holosoma_retargeting.config_types.robot import RobotConfig
robot = RobotConfig(robot_type="g1", robot_dof=23)
print(max(list(robot.MANUAL_UB.keys())))
print(robot._nominal_tracking_indices())
