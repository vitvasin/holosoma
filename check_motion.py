import torch
import numpy as np
from pathlib import Path

npz_path = "src/holosoma/holosoma/data/motions/g1_23dof/whole_body_tracking/sub3_largebox_003_mj.npz"

if not Path(npz_path).exists():
    print(f"File {npz_path} not found.")
else:
    data = np.load(npz_path)
    print("Files in npz:", data.files)
    if "root_translation" in data.files:
        print("Root trans first frame:", data["root_translation"][0])
    if "joint_pos" in data.files:
        print("Joint pos first frame:", data["joint_pos"][0])
    if "body_pos_w" in data.files:
        print("body_pos_w shape:", data["body_pos_w"].shape)
        print("root pos (body 0) first frame:", data["body_pos_w"][0, 0])
