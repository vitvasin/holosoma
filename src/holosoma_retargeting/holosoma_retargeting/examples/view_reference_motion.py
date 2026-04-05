#!/usr/bin/env python3
"""
view_reference_motion.py
Visualize raw human reference motion (before retargeting) using viser.

Usage:
  # LAFAN (dance, walk, run, etc.):
  python examples/view_reference_motion.py --file demo_data/lafan/dance2_subject1.npy --format lafan

  # SMPLH/OMOMO (large box manipulation):
  python examples/view_reference_motion.py --file demo_data/OMOMO_new/sub3_largebox_003.pt --format smplh

  # C3D converted NPZ:
  python examples/view_reference_motion.py --file demo_data/c3d/boxing1.npz --format c3d
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser


# ── Skeleton connectivity ─────────────────────────────────────────────────────

LAFAN_JOINTS = [
    "Hips", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "Spine", "Spine1", "Spine2", "Neck", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
]

LAFAN_BONES = [
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"), ("RightFoot", "RightToeBase"),
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Neck"), ("Neck", "Head"),
    ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
]

SMPLH_JOINTS = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    # finger joints skipped for clarity
]

SMPLH_BONES = [
    ("Pelvis", "L_Hip"), ("L_Hip", "L_Knee"), ("L_Knee", "L_Ankle"), ("L_Ankle", "L_Toe"),
    ("Pelvis", "R_Hip"), ("R_Hip", "R_Knee"), ("R_Knee", "R_Ankle"), ("R_Ankle", "R_Toe"),
    ("Pelvis", "Torso"), ("Torso", "Spine"), ("Spine", "Chest"), ("Chest", "Neck"), ("Neck", "Head"),
    ("Chest", "L_Thorax"), ("L_Thorax", "L_Shoulder"), ("L_Shoulder", "L_Elbow"), ("L_Elbow", "L_Wrist"),
    ("Chest", "R_Thorax"), ("R_Thorax", "R_Shoulder"), ("R_Shoulder", "R_Elbow"), ("R_Elbow", "R_Wrist"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_lafan(path: str) -> tuple[np.ndarray, list[str], list[tuple[str, str]], float]:
    """Load LAFAN .npy file → (T, J, 3) joint positions in world space."""
    data = np.load(path)  # (T, J, 3) in Y-up
    # Convert Y-up → Z-up
    joints = data[..., [0, 2, 1]]
    joints[..., 1] *= -1  # flip Y axis
    return joints, LAFAN_JOINTS, LAFAN_BONES, 30.0


def load_smplh(path: str) -> tuple[np.ndarray, list[str], list[tuple[str, str]], float]:
    """Load SMPLH/OMOMO .pt file → (T, J, 3) joint positions in world space.

    OMOMO .pt files are raw tensors of shape (T, N) where columns 162:318
    encode 52 joint positions (52*3=156 values) in SMPLH format.
    """
    import torch
    data = torch.load(path, map_location="cpu")

    if isinstance(data, torch.Tensor):
        # OMOMO/InterMimic raw format: (T, flat) tensor
        raw = data.detach().numpy()
        # Columns 162..317 → 52 joints × 3
        joints = raw[:, 162 : 162 + 52 * 3].reshape(-1, 52, 3)
    elif isinstance(data, dict):
        # Try common SMPLH dict keys
        for key in ("joints", "smpl_joints", "joint_positions"):
            if key in data:
                joints = data[key].detach().numpy()
                break
        else:
            # Fallback: first 3-D tensor key
            for k, v in data.items():
                if hasattr(v, "detach") and v.ndim == 3:
                    print(f"  Using key '{k}' with shape {v.shape}")
                    joints = v.detach().numpy()
                    break
            else:
                raise ValueError(f"Cannot find joint positions in keys: {list(data.keys())}")
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")

    # Keep only the first 22 body joints (skip fingers)
    joints = joints[:, :len(SMPLH_JOINTS), :]
    return joints, SMPLH_JOINTS, SMPLH_BONES, 30.0

# ── C3D skeleton (matches PIG joints in prep_c3d_for_rt.py) ──────────────────

# Bone connectivity for the 19-joint C3D/PIG skeleton
C3D_BONES = [
    # Spine
    ("Pelvis", "Spine"), ("Spine", "Chest"), ("Chest", "Neck"), ("Neck", "Head"),
    # Left leg
    ("Pelvis", "L_Hip"), ("L_Hip", "L_Knee"), ("L_Knee", "L_Ankle"), ("L_Ankle", "L_Toe"),
    # Right leg
    ("Pelvis", "R_Hip"), ("R_Hip", "R_Knee"), ("R_Knee", "R_Ankle"), ("R_Ankle", "R_Toe"),
    # Left arm
    ("Chest", "L_Shoulder"), ("L_Shoulder", "L_Elbow"), ("L_Elbow", "L_Wrist"),
    # Right arm
    ("Chest", "R_Shoulder"), ("R_Shoulder", "R_Elbow"), ("R_Elbow", "R_Wrist"),
]


def load_c3d(path: str) -> tuple[np.ndarray, list[str], list[tuple[str, str]], float]:
    """Load C3D-converted .npz file produced by prep_c3d_for_rt.py.

    The NPZ must contain:
      - global_joint_positions: (T, J, 3) float32, metres
      - joint_names: array of J joint name strings
      - fps: scalar playback rate
    """
    data = np.load(path, allow_pickle=True)
    joints = data["global_joint_positions"].astype(np.float32)  # (T, J, 3)
    joint_names = list(data["joint_names"])                      # list[str]
    fps = float(data["fps"]) if "fps" in data else 100.0

    # Build bone list — keep only pairs where both joints are present
    name_set = set(joint_names)
    bones = [(a, b) for (a, b) in C3D_BONES if a in name_set and b in name_set]

    print(f"  Joints ({len(joint_names)}): {', '.join(joint_names)}")
    return joints, joint_names, bones, fps



def run_viewer(
    joints: np.ndarray,       # (T, J, 3)
    joint_names: list[str],
    bones: list[tuple[str, str]],
    fps: float,
) -> None:
    T, J, _ = joints.shape
    idx = {name: i for i, name in enumerate(joint_names)}

    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=6, height=6)

    # Pre-create point handles for each joint
    point_handles = {}
    for name in joint_names:
        point_handles[name] = server.scene.add_icosphere(
            f"/human/{name}",
            radius=0.025,
            color=(100, 200, 255),
        )

    # Pre-create line handles for each bone
    bone_handles = {}
    for (a, b) in bones:
        bone_handles[(a, b)] = server.scene.add_spline_catmull_rom(
            f"/bones/{a}_{b}",
            positions=np.zeros((2, 3)),
            color=(255, 180, 50),
            line_width=3,
        )

    # GUI controls
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", 0, T - 1, step=1, initial_value=0)
        playing_cb = server.gui.add_checkbox("Playing", initial_value=True)
        fps_slider = server.gui.add_slider("FPS", 1, 120, step=1, initial_value=int(fps))

    print(f"[viewer] {T} frames | {J} joints | {fps} FPS")
    print("Open http://localhost:8080 in your browser. Press Ctrl+C to exit.")

    frame = 0
    while True:
        if playing_cb.value:
            frame = (frame + 1) % T
            frame_slider.value = frame
        else:
            frame = int(frame_slider.value)

        pos = joints[frame]  # (J, 3)

        # Update joints
        for name in joint_names:
            i = idx[name]
            point_handles[name].position = pos[i]

        # Update bones
        for (a, b) in bones:
            if a in idx and b in idx:
                pts = np.stack([pos[idx[a]], pos[idx[b]]], axis=0)
                bone_handles[(a, b)].positions = pts

        time.sleep(1.0 / max(fps_slider.value, 1))


# ── CLI ───────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    file: str
    """Path to the motion file (.npy for LAFAN, .pt for SMPLH/OMOMO, .npz for C3D)"""
    format: str = "lafan"
    """Data format: 'lafan', 'smplh', or 'c3d'"""
    fps: float = 0.0
    """Override playback FPS (0 = use file default)"""


def main(cfg: Config) -> None:
    path = Path(cfg.file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Loading {cfg.format} motion from: {path}")
    if cfg.format == "lafan":
        joints, joint_names, bones, fps = load_lafan(str(path))
    elif cfg.format in ("smplh", "omomo"):
        joints, joint_names, bones, fps = load_smplh(str(path))
    elif cfg.format == "c3d":
        joints, joint_names, bones, fps = load_c3d(str(path))
    else:
        raise ValueError(f"Unknown format '{cfg.format}'. Use: lafan, smplh, c3d")

    if cfg.fps > 0:
        fps = cfg.fps

    print(f"  Shape: {joints.shape}  FPS: {fps}")
    run_viewer(joints, joint_names, bones, fps)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
