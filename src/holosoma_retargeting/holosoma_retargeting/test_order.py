from holosoma_retargeting.config_types.data_conversion import DataConversionConfig
from holosoma_retargeting.config_types.robot import RobotConfig
import mujoco

cfg = DataConversionConfig(
    input_file="test",
    robot="g1",
    robot_config=RobotConfig(robot_type="g1", robot_dof=23)
)
joint_names = cfg.JOINT_NAMES

m = mujoco.MjModel.from_xml_path("../../models/g1/g1_23dof.urdf")
urdf_joints = []
for i in range(m.njnt):
    if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
        continue
    urdf_joints.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))

print("Is order equal?", joint_names == urdf_joints)
print("Differences:")
for j1, j2 in zip(joint_names, urdf_joints):
    if j1 != j2:
        print(f"Hardcoded: {j1} <-> URDF: {j2}")
