#!/usr/bin/env python3
"""
prep_c3d_for_rt.py
Convert .c3d motion capture files to the .npz format required by the retargeting pipeline.

Supports the Plug-In Gait (PIG) marker set (standard Vicon/Codamotion output).
For other marker sets, use --marker-set custom and provide a --marker-map JSON file.

Usage:
  # Single file (PIG marker set):
  python data_utils/prep_c3d_for_rt.py \
      --input /path/to/motion.c3d \
      --output demo_data/c3d/motion_name.npz

  # Whole directory:
  python data_utils/prep_c3d_for_rt.py \
      --input /path/to/c3d_folder/ \
      --output demo_data/c3d/

  # Custom marker set (provide JSON mapping):
  python data_utils/prep_c3d_for_rt.py \
      --input motion.c3d \
      --output out.npz \
      --marker-set custom \
      --marker-map my_marker_map.json

Install dependencies first:
  pip install ezc3d scipy numpy
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

try:
    import ezc3d
except ImportError:
    print("ERROR: ezc3d not installed. Run: pip install ezc3d")
    sys.exit(1)

try:
    from scipy.interpolate import interp1d
    from scipy.signal import butter, filtfilt
except ImportError:
    print("ERROR: scipy not installed. Run: pip install scipy")
    sys.exit(1)


# ── Plug-In Gait (PIG) marker set ────────────────────────────────────────────
# Defines how to compute each skeleton joint from PIG markers.
# Each entry: joint_name → list of marker names to AVERAGE.
# If any marker is missing, all available ones are used.

PIG_JOINT_FROM_MARKERS: dict[str, list[str]] = {
    # Core
    "Pelvis":      ["LASI", "RASI", "LPSI", "RPSI"],
    "L_Hip":       ["LASI", "LPSI"],
    "R_Hip":       ["RASI", "RPSI"],
    "L_Knee":      ["LKNE"],
    "R_Knee":      ["RKNE"],
    "L_Ankle":     ["LANK"],
    "R_Ankle":     ["RANK"],
    "L_Toe":       ["LTOE", "LTOEBASE"],
    "R_Toe":       ["RTOE", "RTOEBASE"],
    # Spine / torso
    "Spine":       ["STRN", "C7", "T10"],
    "Chest":       ["LSHO", "RSHO"],
    "Neck":        ["C7"],
    "Head":        ["RFHD", "LFHD", "RBHD", "LBHD"],
    # Arms
    "L_Shoulder":  ["LSHO"],
    "R_Shoulder":  ["RSHO"],
    "L_Elbow":     ["LELB"],
    "R_Elbow":     ["RELB"],
    "L_Wrist":     ["LWRA", "LWRB"],
    "R_Wrist":     ["RWRA", "RWRB"],
}

# Standard joint order for the "c3d" format (matches retargeter expectations)
C3D_DEMO_JOINTS = list(PIG_JOINT_FROM_MARKERS.keys())

# Height contributors: use the head markers to estimate subject height
HEIGHT_MARKERS = ["RFHD", "LFHD", "RBHD", "LBHD", "C7"]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _get_marker_data(c3d_file: ezc3d.c3d, name: str) -> np.ndarray | None:
    """Return (T, 3) array for a marker name (mm → m), or None if absent.

    Handles subject-prefixed labels (e.g. 'rory7:LASI') by also matching
    on the suffix after the colon separator.
    """
    labels = c3d_file["parameters"]["POINT"]["LABELS"]["value"]
    name_up = name.upper()

    idx = None
    # 1) Exact match (case-insensitive)
    idx = next((i for i, lbl in enumerate(labels) if lbl.strip().upper() == name_up), None)
    # 2) Suffix match: handles 'SubjectName:MARKERNAME' prefixes
    if idx is None:
        idx = next(
            (i for i, lbl in enumerate(labels)
             if lbl.strip().upper().split(":")[-1] == name_up),
            None,
        )
    if idx is None:
        return None
    pts = c3d_file["data"]["points"][:3, idx, :].T  # (T, 3)
    pts = pts / 1000.0  # mm → m
    # Mask out invalid frames (residual == 0 in c3d convention → marker occluded)
    residuals = c3d_file["data"]["points"][3, idx, :]
    # Some c3d files use 0 residual for missing; others use -1. Handle both.
    invalid = (residuals <= 0) | np.any(pts == 0.0, axis=1)
    pts[invalid] = np.nan
    return pts


def _fill_gaps(data: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """Linear interpolate small NaN gaps in (T, 3) marker data."""
    out = data.copy()
    for dim in range(3):
        col = data[:, dim]
        valid_idx = np.where(~np.isnan(col))[0]
        if len(valid_idx) < 2:
            continue
        nan_idx = np.where(np.isnan(col))[0]
        # Only fill gaps smaller than max_gap
        f = interp1d(valid_idx, col[valid_idx], kind="linear",
                     bounds_error=False, fill_value=np.nan)
        filled = f(np.arange(len(col)))
        # Only apply fill to small gaps
        for i in nan_idx:
            # Find gap extent
            left = valid_idx[valid_idx < i]
            right = valid_idx[valid_idx > i]
            if len(left) > 0 and len(right) > 0:
                gap = right[0] - left[-1]
                if gap <= max_gap:
                    out[i, dim] = filled[i]
    return out


def _lowpass_filter(data: np.ndarray, fps: float, cutoff_hz: float = 6.0) -> np.ndarray:
    """Apply 4th-order Butterworth low-pass filter to (T, 3) data."""
    if fps <= 0 or cutoff_hz <= 0:
        return data
    nyq = fps / 2.0
    if cutoff_hz >= nyq:
        return data
    b, a = butter(4, cutoff_hz / nyq, btype="low")
    out = data.copy()
    for dim in range(3):
        col = data[:, dim]
        valid = ~np.isnan(col)
        if valid.sum() > 10:
            out[valid, dim] = filtfilt(b, a, col[valid])
    return out


def compute_joint_positions(
    c3d_file: ezc3d.c3d,
    joint_map: dict[str, list[str]],
    fps: float,
    lowpass_hz: float,
    verbose: bool,
) -> np.ndarray:
    """
    Build (T, J, 3) global joint positions from C3D markers.
    Missing entire joints are forward/backward filled from neighbours.
    """
    T = c3d_file["data"]["points"].shape[2]
    J = len(joint_map)
    joints = np.full((T, J, 3), np.nan)

    for j_idx, (joint_name, marker_names) in enumerate(joint_map.items()):
        parts = []
        for mname in marker_names:
            d = _get_marker_data(c3d_file, mname)
            if d is not None:
                parts.append(d)
            elif verbose:
                print(f"  [warn] Marker '{mname}' not found (joint '{joint_name}')")

        if not parts:
            if verbose:
                print(f"  [warn] No markers found for joint '{joint_name}' — will be filled")
            continue

        # Average available markers
        stacked = np.stack(parts, axis=0)  # (N_markers, T, 3)
        with warnings_suppressed():
            avg = np.nanmean(stacked, axis=0)  # (T, 3)

        avg = _fill_gaps(avg)
        avg = _lowpass_filter(avg, fps, lowpass_hz)
        joints[:, j_idx, :] = avg

    # Forward/backward fill any fully-missing joints using nearest valid joint
    for j_idx in range(J):
        col = joints[:, j_idx, :]
        if np.all(np.isnan(col)):
            # Try to impute from the average of all other joints
            other = np.nanmean(joints, axis=1)  # (T, 3)
            joints[:, j_idx, :] = other

    return joints


class warnings_suppressed:
    """Context manager to suppress numpy runtime warnings."""
    def __enter__(self):
        self._old = np.seterr(all="ignore")
    def __exit__(self, *_):
        np.seterr(**self._old)


def estimate_height(c3d_file: ezc3d.c3d) -> float:
    """Estimate subject height (m) from head-marker peak height + ground offset."""
    tallest = 0.0
    for mname in HEIGHT_MARKERS:
        d = _get_marker_data(c3d_file, mname)
        if d is not None:
            h = np.nanmax(d[:, 2])
            tallest = max(tallest, h)
    if tallest < 0.5:
        print("  [warn] Could not estimate height from markers, defaulting to 1.75 m")
        return 1.75
    # Add ~15 cm for the top of the skull (head centre to crown)
    return tallest + 0.15


def convert_c3d_to_npz(
    c3d_path: Path,
    output_path: Path,
    joint_map: dict[str, list[str]],
    downsample_to: int | None,
    lowpass_hz: float,
    verbose: bool,
) -> None:
    if verbose:
        print(f"\n[convert] {c3d_path.name}")

    c3d = ezc3d.c3d(str(c3d_path))
    fps_orig = c3d["parameters"]["POINT"]["RATE"]["value"][0]
    T_orig = c3d["data"]["points"].shape[2]

    if verbose:
        print(f"  FPS: {fps_orig}  Frames: {T_orig}")
        labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
        print(f"  Markers ({len(labels)}): {', '.join(l.strip() for l in labels[:10])}{'...' if len(labels)>10 else ''}")

    joints = compute_joint_positions(c3d, joint_map, fps_orig, lowpass_hz, verbose)

    # Downsample
    fps_out = fps_orig
    if downsample_to is not None and downsample_to < fps_orig:
        step = max(1, round(fps_orig / downsample_to))
        joints = joints[::step]
        fps_out = fps_orig / step
        if verbose:
            print(f"  Downsampled: {fps_orig} → {fps_out:.1f} Hz  Frames: {joints.shape[0]}")

    height = estimate_height(c3d)
    if verbose:
        print(f"  Estimated height: {height:.3f} m")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        global_joint_positions=joints.astype(np.float32),
        height=np.float32(height),
        fps=np.float32(fps_out),
        joint_names=np.array(list(joint_map.keys())),
        source_file=np.array(str(c3d_path)),
    )
    print(f"  Saved → {output_path}  shape={joints.shape}")


# ── CLI ───────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    input: str
    """Path to a .c3d file OR a directory of .c3d files."""

    output: str
    """Output .npz file path, or output directory (when input is a directory)."""

    marker_set: str = "pig"
    """Marker set to use: 'pig' (Plug-In Gait, default) or 'custom'."""

    marker_map: str | None = None
    """Path to a JSON file defining {joint_name: [marker1, marker2, ...]} (required for --marker-set custom)."""

    downsample_to: int | None = 100
    """Target output frame rate in Hz. Set None to keep original. Default 100 Hz."""

    lowpass_hz: float = 6.0
    """Low-pass filter cutoff frequency in Hz. Set 0 to skip filtering."""

    verbose: bool = True
    """Print per-file details."""


def main(cfg: Config) -> None:
    # Load marker map
    if cfg.marker_set == "pig":
        joint_map = PIG_JOINT_FROM_MARKERS
        print("Using Plug-In Gait (PIG) marker set.")
    elif cfg.marker_set == "custom":
        if cfg.marker_map is None:
            print("ERROR: --marker-map is required when --marker-set custom")
            sys.exit(1)
        with open(cfg.marker_map) as f:
            joint_map = json.load(f)
        print(f"Using custom marker map from: {cfg.marker_map}")
    else:
        print(f"ERROR: Unknown marker set '{cfg.marker_set}'. Use 'pig' or 'custom'.")
        sys.exit(1)

    input_path = Path(cfg.input)
    output_path = Path(cfg.output)

    if input_path.is_dir():
        c3d_files = sorted(input_path.glob("*.c3d")) + sorted(input_path.glob("*.C3D"))
        if not c3d_files:
            print(f"No .c3d files found in {input_path}")
            sys.exit(1)
        output_path.mkdir(parents=True, exist_ok=True)
        for c3d_path in c3d_files:
            out = output_path / (c3d_path.stem + ".npz")
            try:
                convert_c3d_to_npz(c3d_path, out, joint_map,
                                   cfg.downsample_to, cfg.lowpass_hz, cfg.verbose)
            except Exception as e:
                print(f"  [error] {c3d_path.name}: {e}")
    elif input_path.suffix.lower() == ".c3d":
        if output_path.is_dir() or not output_path.suffix:
            output_path = output_path / (input_path.stem + ".npz")
        convert_c3d_to_npz(input_path, output_path, joint_map,
                           cfg.downsample_to, cfg.lowpass_hz, cfg.verbose)
    else:
        print(f"ERROR: Input must be a .c3d file or a directory. Got: {input_path}")
        sys.exit(1)

    print("\nDone! Next steps:")
    print("  1. Run retargeting:")
    print(f"     python examples/robot_retarget.py \\")
    print(f"         --robot-config.robot-dof 23 \\")
    print(f"         --data_path {output_path if input_path.is_dir() else output_path.parent} \\")
    print(f"         --task-name {input_path.stem if input_path.is_file() else '<sequence_name>'} \\")
    print(f"         --data_format c3d")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
