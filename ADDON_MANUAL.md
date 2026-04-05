# Holosoma Add-On Manual

> This document covers all additions made to the Holosoma framework beyond the original codebase.
> It is automatically updated as new features are added.

---

## Table of Contents

1. [G1-23DOF Robot Support](#1-g1-23dof-robot-support)
2. [Demo Scripts](#2-demo-scripts)
3. [Reference Motion Viewer](#3-reference-motion-viewer)
4. [C3D MoCap Format Support](#4-c3d-mocap-format-support)
5. [Training Optimization Guide](#5-training-optimization-guide)
6. [Available Datasets & Tasks](#6-available-datasets--tasks)
7. [Inference & Evaluation](#7-inference--evaluation)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. G1-23DOF Robot Support

The Unitree G1 robot is available in two variants:

| Variant | DOF | Config key | Notes |
|---------|-----|-----------|-------|
| G1-29DOF | 29 | `g1_29dof` | Original, includes waist roll/pitch and wrist yaw |
| **G1-23DOF** | **23** | `g1_23dof` | **New** - no waist roll/pitch, rubber hand end-effectors |

### Modified Files

| File | Change |
|------|--------|
| `src/holosoma_retargeting/holosoma_retargeting/config_types/robot.py` | Added 23-DOF conditional logic and joint filtering |
| `src/holosoma_retargeting/holosoma_retargeting/config_types/data_conversion.py` | Dynamic joint name filtering for 23-DOF |
| `src/holosoma/holosoma/config_values/robot.py` | New `g1_23dof` and `g1_23dof_w_object` configs with corrected body names |
| `src/holosoma/holosoma/config_values/action.py` | Added `g1_23dof_joint_pos` action config |
| `src/holosoma/holosoma/config_values/experiment.py` | Registered 23-DOF experiment presets |
| `src/holosoma/holosoma/config_values/wbt/g1_23dof/command.py` | Fixed body names to match 23-DOF URDF |
| `src/holosoma/holosoma/config_values/wbt/g1_23dof/termination.py` | Fixed body names to match 23-DOF URDF |

### Key Design Differences from 29-DOF

| Property | 29-DOF | 23-DOF |
|----------|--------|--------|
| Waist joint | `waist_yaw_link` (active) | `torso_link` (fixed) |
| Wrist terminal body | `left/right_wrist_roll_link` | `left/right_wrist_roll_rubber_hand` |
| Foot contact body | `left/right_foot_contact_point` (virtual) | `left/right_ankle_roll_link` |
| Foot height reference | `foot_contact_point` | `ankle_roll_link` |
| `num_bodies` | 28 | 24 |

### Training Experiment Presets

| Preset | Algorithm | Description |
|--------|-----------|-------------|
| `exp:g1-23dof-wbt` | PPO | Whole-body tracking |
| `exp:g1-23dof-wbt-fast-sac` | FastSAC | Faster convergence (recommended) |
| `exp:g1-23dof-wbt-w-object` | PPO | With object interaction |
| `exp:g1-23dof-wbt-fast-sac-w-object` | FastSAC | With object + faster convergence |

---

## 2. Demo Scripts

All demo scripts are in `demo_scripts/` and source the correct conda environment automatically.

### 2.1 OMOMO/SMPLH Dataset

**Script:** `demo_scripts/demo_g1_23dof_wb_tracking.sh`

```bash
./demo_scripts/demo_g1_23dof_wb_tracking.sh
```

**Available tasks** (edit `--task-name` on lines 53 and 57):

| Task name | Description |
|-----------|-------------|
| `sub3_largebox_003` | Subject 3 lifting large box *(default)* |
| `sub10_largebox_049` | Subject 10 lifting large box |

**Data location:** `src/holosoma_retargeting/holosoma_retargeting/demo_data/OMOMO_new/`

---

### 2.2 LAFAN Dataset

**Script:** `demo_scripts/demo_g1_23dof_lafan_wb_tracking.sh`

```bash
./demo_scripts/demo_g1_23dof_lafan_wb_tracking.sh
```

Downloads and processes the Ubisoft LAFAN1 dataset automatically on first run (~700 MB, ~10 min).

**Available tasks** (edit `--task-name` on line 120):

| Category | Example task names |
|----------|--------------------|
| Walking | `walk1_subject1`, `walk2_subject1`, `walk3_subject1` (x5 subjects) |
| Running | `run1_subject2`, `run2_subject1`, `sprint1_subject2` |
| Dancing | `dance1_subject1`, `dance2_subject1` *(default)* |
| Jumping | `jumps1_subject1`, `jumps1_subject2` |
| Ground | `ground1_subject1`, `ground2_subject2` |
| Fall & Get Up | `fallAndGetUp1_subject1`, `fallAndGetUp2_subject2` |
| Obstacles | `obstacles1_subject1` ... `obstacles6_subject5` |
| Fight/Push | `fight1_subject2`, `push1_subject2`, `pushAndFall1_subject1` |
| Aiming | `aiming1_subject1`, `aiming2_subject2` |
| Multi-action | `multipleActions1_subject1` ... `multipleActions1_subject4` |

**Data location:** `src/holosoma_retargeting/holosoma_retargeting/demo_data/lafan/`

---

### 2.3 C3D MoCap Data

**Script:** `demo_scripts/demo_g1_23dof_c3d_wb_tracking.sh`

```bash
# Pass your .c3d file as the first argument:
./demo_scripts/demo_g1_23dof_c3d_wb_tracking.sh /path/to/motion.c3d [sequence_name]

# Example:
./demo_scripts/demo_g1_23dof_c3d_wb_tracking.sh /data/mocap/walk1.c3d walk1
```

Requires a `.c3d` file. Supports Plug-In Gait (PIG) and multi-subject prefixed marker names (e.g. `rory7:LASI`).
See [Section 4](#4-c3d-mocap-format-support) for non-PIG marker sets.

**Step 2 & 3 skip prompts:** If retargeted or converted files already exist, the script asks whether to skip or overwrite them. Press Enter to skip (default), or type `n` to reprocess.

---

### Training Flags Reference

All demo scripts share these configurable training flags (edit bottom of each script):

| Flag | Current value | Description |
|------|------|-------------|
| `--training.headless` | `True` | Set `False` for live Isaac Sim viewport |
| `--training.num-envs` | `4096` | Parallel environments |
| `--algo.config.num-learning-iterations` | `50000` | Total training steps |
| `--simulator.config.sim.max-episode-length-s` | `6.0` | Episode duration in seconds |
| `--logger.video.enabled` | `True` | Save video clips to disk |
| `--logger.video.interval` | `5` | Record every N episodes |
| `--logger.video.save-dir` | `logs/videos/...` | Video output directory |

**Videos** are saved to `holosoma/logs/videos/<script-name>/`.

---

## 3. Reference Motion Viewer

**Script:** `src/holosoma_retargeting/holosoma_retargeting/examples/view_reference_motion.py`

Visualizes the raw human motion dataset as a 3D stick-figure skeleton in a web browser — useful for verifying data quality **before retargeting**.

### Usage

```bash
# Activate the retargeting environment first:
source scripts/source_retargeting_setup.sh
cd src/holosoma_retargeting/holosoma_retargeting

# View OMOMO/SMPLH data:
python examples/view_reference_motion.py \
    --file demo_data/OMOMO_new/sub3_largebox_003.pt \
    --format smplh

# View LAFAN data:
python examples/view_reference_motion.py \
    --file demo_data/lafan/dance2_subject1.npy \
    --format lafan

# View C3D converted NPZ:
python examples/view_reference_motion.py \
    --file demo_data/c3d/boxing1.npz \
    --format c3d

# Open in browser: http://localhost:8080
```

### Supported Formats

| `--format` | File type | Dataset |
|------------|-----------|---------|
| `smplh` | `.pt` | OMOMO/InterMimic (raw tensor, columns 162-318) |
| `lafan` | `.npy` | LAFAN1 (Y-up, auto-converted to Z-up) |
| `c3d` | `.npz` | C3D converted by `prep_c3d_for_rt.py` (PIG skeleton) |

### Browser Controls

| Control | Description |
|---------|-------------|
| **Frame** slider | Scrub through animation frames |
| **Playing** checkbox | Play / Pause |
| **FPS** slider | Adjust playback speed (1-120) |
| Mouse drag | Rotate view |
| Scroll wheel | Zoom |

---

## 4. C3D MoCap Format Support

Support for `.c3d` files (Vicon, Codamotion, Qualisys, etc.) using the Plug-In Gait (PIG) marker set.

### Converter Script

**Script:** `src/holosoma_retargeting/holosoma_retargeting/data_utils/prep_c3d_for_rt.py`

```bash
source scripts/source_retargeting_setup.sh
pip install ezc3d  # one-time install
cd src/holosoma_retargeting/holosoma_retargeting

# Single file:
python data_utils/prep_c3d_for_rt.py \
    --input /path/to/motion.c3d \
    --output demo_data/c3d/motion.npz

# Whole directory:
python data_utils/prep_c3d_for_rt.py \
    --input /path/to/c3d_folder/ \
    --output demo_data/c3d/
```

### Converter Options

| Flag | Default | Description |
|------|---------|-------------|
| `--marker-set` | `pig` | Marker set: `pig` (auto) or `custom` (with JSON map) |
| `--marker-map` | None | Path to JSON file for custom marker sets |
| `--downsample-to` | `100` | Output framerate in Hz |
| `--lowpass-hz` | `6.0` | Butterworth low-pass cutoff (0 = skip filtering) |

### PIG Marker to Skeleton Joint Mapping

| Skeleton Joint | PIG Markers Used |
|---------------|-----------------|
| `Pelvis` | LASI, RASI, LPSI, RPSI (averaged) |
| `L_Hip` / `R_Hip` | LASI+LPSI / RASI+RPSI |
| `L_Knee` / `R_Knee` | LKNE / RKNE |
| `L_Ankle` / `R_Ankle` | LANK / RANK |
| `L_Toe` / `R_Toe` | LTOE / RTOE |
| `Spine` | STRN or C7 or T10 |
| `Chest` | LSHO + RSHO (averaged) |
| `L_Shoulder` / `R_Shoulder` | LSHO / RSHO |
| `L_Elbow` / `R_Elbow` | LELB / RELB |
| `L_Wrist` / `R_Wrist` | LWRA+LWRB / RWRA+RWRB (averaged) |
| `Head` | RFHD+LFHD+RBHD+LBHD (averaged) |

### Custom Marker Sets

```bash
# Create a JSON mapping (joint_name -> list of markers):
cat > my_markers.json << 'EOF'
{
    "Pelvis":     ["SACR", "LASI", "RASI"],
    "L_Knee":     ["LKNE"],
    "R_Knee":     ["RKNE"],
    "L_Ankle":    ["LANK"],
    "R_Ankle":    ["RANK"],
    "L_Shoulder": ["LSHO"],
    "R_Shoulder": ["RSHO"]
}
EOF

python data_utils/prep_c3d_for_rt.py \
    --input motion.c3d \
    --output out.npz \
    --marker-set custom \
    --marker-map my_markers.json
```

### Converting From Other Formats to C3D

| Source Format | Tool |
|--------------|------|
| `.tvd` (Vicon) | Export from Vicon Nexus: File > Export > C3D |
| `.amc` + `.asf` (CMU) | `pip install pymo` |
| `.bvh` | `pip install bvhtoolbox` (`bvh2c3d`) |
| `.trc` (OpenSim) | OpenSim GUI: Tools > Export |

### Format Registration in Framework

The `c3d` format is registered in:
```
src/holosoma_retargeting/holosoma_retargeting/config_types/data_type.py
```

Entries added:
- `C3D_DEMO_JOINTS` constant (19 joints)
- `DEMO_JOINTS_REGISTRY["c3d"]`
- `TOE_NAMES_BY_FORMAT["c3d"]`
- `JOINTS_MAPPINGS[("c3d", "g1")]` and `[("c3d", "t1")]`

---

## 5. Training Optimization Guide

### Algorithm Comparison

| Algorithm | Preset | Iterations needed | GPU usage | Notes |
|-----------|--------|-------------------|-----------|-------|
| **FastSAC** | `exp:g1-23dof-wbt-fast-sac` | ~50,000 | Higher (replay buffer) | Recommended |
| PPO | `exp:g1-23dof-wbt` | ~30,000 | Lower | Stable baseline |

FastSAC advantages: AMP (`bf16`), `torch.compile` kernel fusion, off-policy replay.

### Environment Count vs. VRAM (FastSAC)

| `--training.num-envs` | Est. VRAM | Recommendation |
|----------------------|-----------|---------------|
| 1024 | ~6 GB | Debug only |
| 2048 | ~9 GB | Low VRAM |
| **4096** | **~14 GB** | **Default (safe for 24 GB GPU)** |
| 8192 | ~20 GB | High VRAM systems |
| 10240+ | >22 GB | OOM risk on 24 GB |

Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already in all demo scripts).

### Reducing Training Time

In order of impact:
1. **Switch to FastSAC** (`exp:g1-23dof-wbt-fast-sac`) — 3-5x faster than PPO
2. **Shorten episodes** (`--simulator.config.sim.max-episode-length-s 6.0`) — 1.5x faster
3. **Reduce iterations** (`--algo.config.num-learning-iterations 8000`) — for quick validation
4. **Check GPU utilization** (`nvidia-smi dmon`) and scale `num-envs` accordingly

### Live Visualization During Training

```bash
# Add these flags to any demo script for a GUI window:
--training.headless False \
--training.num-envs 64     # Reduce envs for smooth rendering
```

---

## 6. Available Datasets & Tasks

### Quick Reference

| Dataset | Format | # Tasks | Notes |
|---------|--------|---------|-------|
| OMOMO | `smplh` | 2 | Box manipulation |
| LAFAN1 | `lafan` | 75 | General locomotion + sports |
| Custom C3D | `c3d` | Unlimited | Your lab data |
| AMASS SMPLX | `smplx` | Many | See original framework docs |

### Adding a New Dataset Format

Edit **only** `config_types/data_type.py` (see `ADD_MOTION_FORMAT_README.md` for full guide):

1. Define `MYFORMAT_DEMO_JOINTS = [...]`
2. Add to `DEMO_JOINTS_REGISTRY`
3. Add to `TOE_NAMES_BY_FORMAT`
4. Add to `JOINTS_MAPPINGS` for each robot type

---

## 7. Inference & Evaluation

After training completes, two modes are available depending on whether you have real hardware.

### Inference Environments

| Mode | Conda env | Script |
|------|-----------|--------|
| **Simulation** (IsaacSim) | `hssim` | `eval_agent.py` |
| **Real robot** (hardware) | `hsinference` | `run_policy.py` |

> **Important:** `run_policy.py` is for the **physical G1 robot** only. It communicates via DDS/ZMQ and crashes without hardware present.

---

### 7.1 Simulation Evaluation (IsaacSim)

**Script:** `demo_scripts/demo_g1_23dof_inference.sh` (default `--sim` mode)

```bash
# Default: opens IsaacSim with the trained policy
./demo_scripts/demo_g1_23dof_inference.sh \
    --checkpoint logs/WholeBodyTracking/<run-dir>/model_0050000.pt

# Headless (no GUI window):
./demo_scripts/demo_g1_23dof_inference.sh \
    --checkpoint logs/WholeBodyTracking/<run-dir>/model_0050000.pt \
    --headless
```

Or run directly:
```bash
source scripts/source_isaacsim_setup.sh
pip install -e src/holosoma -q

python src/holosoma/holosoma/eval_agent.py \
    --checkpoint logs/WholeBodyTracking/<run-dir>/model_0050000.pt \
    --training.export-onnx True \
    --training.num-envs 1 \
    --training.headless False
```

This also **exports an ONNX** to `<checkpoint_dir>/exported/<name>.onnx` for later hardware deployment.

**Finding your checkpoint:**
```bash
ls -t logs/WholeBodyTracking/*/model_*.pt | head -5
```

---

### 7.2 Real Robot Hardware Deployment

**Script:** `demo_scripts/demo_g1_23dof_inference.sh --hardware`

```bash
./demo_scripts/demo_g1_23dof_inference.sh \
    --onnx logs/WholeBodyTracking/<run-dir>/exported/model_0050000.onnx \
    --hardware
```

Or run directly:
```bash
source scripts/source_inference_setup.sh
pip install -e src/holosoma_inference -q

python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-23dof-wbt \
    --task.model-path /path/to/model_0050000.onnx
```

**Registered inference presets:**

| Preset | Robot | Use case |
|--------|-------|----------|
| `inference:g1-29dof-loco` | G1-29DOF | Locomotion |
| `inference:g1-29dof-wbt` | G1-29DOF | Whole-body tracking |
| `inference:g1-23dof-wbt` | **G1-23DOF** | **Whole-body tracking (new)** |
| `inference:t1-29dof-loco` | T1-29DOF | Locomotion |

**Keyboard controls** (keep terminal focused):

| Key | Action |
|-----|--------|
| `]` | Start policy |
| `o` | Stop policy |
| `i` | Default pose |
| `s` | Start motion clip |

---

### 7.3 ONNX Export Only

To export without running the full simulation:
```bash
source scripts/source_isaacsim_setup.sh

python src/holosoma/holosoma/eval_agent.py \
    --checkpoint logs/WholeBodyTracking/<run-dir>/model_0050000.pt \
    --training.export-onnx True \
    --training.num-envs 1 \
    --training.headless True
# Press Ctrl+C after "Exported policy as onnx to: ..." appears
```

ONNX is saved to: `logs/WholeBodyTracking/<run-dir>/exported/model_0050000.onnx`

---

## 8. Troubleshooting

### Training Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: module has no attribute 'g1_23dof_joint_pos'` | Stale install | `pip install -e src/holosoma` |
| `ValueError: Not all regular expressions matched` | Body name in `body_names` not in URDF | Check `config_values/robot.py` body_names |
| `AssertionError: specified name ... doesn't exist` | Body name in command/termination not in motion data | Check `wbt/g1_23dof/command.py` body_names_to_track |
| `Unrecognized options: logger:tensorboard` | No tensorboard logger in this version | Use `logger:wandb-offline` |
| `CUDA out of memory` | Too many environments for available VRAM | Reduce `--training.num-envs` |
| `MailboxClosedError` / wandb panic | W&B credential file missing | Use `logger:wandb-offline` |
| `IndexError` in data conversion | Missing `--robot-config.robot-dof 23` | Add flag to conversion command |

### C3D Conversion Errors

| Error | Cause | Fix |
|-------|-------|-----|
| All markers `not found` warnings | Prefixed marker names (e.g. `rory7:LKNE`) | Already handled — suffix matching auto-detects prefixes |
| `RuntimeWarning: Mean of empty slice` | All joints are NaN (no markers matched at all) | Verify marker set; use `--marker-set custom` with JSON map |
| `SVD did not converge` | NaN joint positions fed to retargeter | Fix marker matching first — this is a downstream symptom |

### Viewer Errors

| Error | Fix |
|-------|-----|
| `'list' object has no attribute 'shape'` | Pass `np.ndarray` to viser (fixed in current version) |
| `Can't call numpy() on Tensor that requires grad` | Use `.detach().numpy()` (fixed in current version) |
| Port 8080 in use | `pkill -f viser` |

### Inference Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'holosoma_inference'` | Wrong conda env or package not installed | `source scripts/source_inference_setup.sh && pip install -e src/holosoma_inference` |
| `free(): invalid pointer` / `Aborted (core dumped)` after clock subscriber | `run_policy.py` needs real hardware — DDS/ZMQ crashes without it | Use IsaacSim eval instead: `eval_agent.py --training.headless False` |
| `Model path: ['model.onnx', '--task.simulator', 'mujoco']` | `--task.simulator` is not a valid flag for `run_policy.py` | Remove `--task.simulator mujoco`; use `eval_agent.py` for simulation |
| `eval_agent.py` hangs after `Exported policy as onnx to: ...` | It continues running full IsaacSim eval after export | Press Ctrl+C — export is already done |

### Video Not Found After Training

Videos are buried in the wandb run directory by default:
```
logs/WholeBodyTracking/<run-id>/.wandb/wandb/offline-run-*/files/media/videos/
```

All updated demo scripts save to a simpler location:
```
logs/videos/<script-name>/Training rollout_*.mp4
```

---

*Last updated: 2026-04-03 — Added Section 7: Inference & Evaluation*
