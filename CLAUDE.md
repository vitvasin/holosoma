# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Holosoma is a humanoid robotics framework for training and deploying whole-body tracking and locomotion RL policies. It is a monorepo with three packages under `src/`:

| Package | Purpose |
|---|---|
| `holosoma` | Training framework (PPO, FastSAC, multi-simulator) |
| `holosoma_inference` | Deployment/inference on real robots and MuJoCo |
| `holosoma_retargeting` | Converting human motion capture data to robot motion |

## Install

Each package is installed independently. Do **not** install `holosoma[unitree,booster]` unless deploying to real hardware — those extras pull hardware SDK wheels from GitHub and fail without internet.

```bash
pip install -e src/holosoma
pip install -e src/holosoma_inference
pip install -e src/holosoma_retargeting
# dev extras for retargeting:
pip install -e 'src/holosoma_retargeting[dev]'
```

> **Note:** `holosoma_retargeting` pins `numpy==2.3.5` due to `yourdfpy` compatibility. Do not upgrade numpy in that environment.

## Lint & Type Check

```bash
# Ruff lint (configured in root pyproject.toml)
ruff check .
ruff format .

# Type check (excludes holosoma_inference by default — see mypy.ini)
mypy .

# Pre-commit (runs ruff, ruff-format, mypy, clang-format)
pre-commit run --files <changed files>
```

## Tests

```bash
# Unit tests — exclude IsaacSim and inference-specific tests
pytest -s --ignore=thirdparty --ignore=src/holosoma_inference -m "not isaacsim and not requires_inference"

# Inference tests only
pytest -s src/holosoma_inference/

# Single test file
pytest -s src/holosoma/tests/test_file_cache.py
```

Custom markers: `isaacsim`, `multi_gpu`, `requires_inference` (defined in `conftest.py`).

## Environment Setup

Each simulator has a setup script and a corresponding source script. Always **source** (not run) the source scripts to set environment variables:

```bash
# IsaacSim (whole-body tracking training)
source scripts/source_isaacsim_setup.sh

# IsaacGym (fast locomotion training)
source scripts/source_isaacgym_setup.sh

# MJWarp / MuJoCo (GPU-accelerated MuJoCo training)
source scripts/source_mujoco_setup.sh

# Retargeting pipeline
source scripts/source_retargeting_setup.sh

# Inference / deployment
source scripts/source_inference_setup.sh
```

## Configuration System

All three packages use **Tyro** for config composition. Arguments are composed from registered entry points using `exp:`, `simulator:`, and `logger:` markers:

```bash
python src/holosoma/holosoma/train_agent.py \
    exp:g1-23dof-wbt-fast-sac \
    simulator:isaacsim \
    logger:wandb-offline \
    --training.num-envs 4096 \
    --algo.config.num-learning-iterations 50000
```

Config types live in `config_types/` and default values in `config_values/` within each package. Entry points are registered in `setup.py`.

## Key Entry Points

```bash
# Training
python src/holosoma/holosoma/train_agent.py exp:<name> simulator:<sim> logger:<logger> [overrides]

# Evaluation
python src/holosoma/holosoma/eval_agent.py [args]

# Inference (sim or real robot)
python src/holosoma_inference/holosoma_inference/run_policy.py inference:<config> --task.model-path <path>

# Retarget motion capture to robot NPZ
python src/holosoma_retargeting/holosoma_retargeting/examples/robot_retarget.py \
    --robot-config.robot-dof 23 --data_path <dir> --task-name <name> --data_format lafan

# Convert retargeted NPZ to MuJoCo training format
python src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py \
    --robot-config.robot-dof 23 --input_file <npz> --output_fps 50 --output_name <out> \
    --data_format lafan --object_name ground --once [--line-range start end]

# Visualise motion
python src/holosoma_retargeting/holosoma_retargeting/viser_player.py [args]
```

## Architecture

### Training Pipeline (`holosoma`)

```
train_agent.py
  └── ExperimentConfig (Tyro)
        ├── Simulator  (IsaacGym | IsaacSim | MJWarp)  ← src/holosoma/holosoma/simulator/
        ├── Environment (locomotion | whole-body-tracking) ← src/holosoma/holosoma/envs/
        ├── Agent      (PPO | FastSAC)                  ← src/holosoma/holosoma/agents/
        └── Logger     (wandb | wandb-offline | disabled)
```

- **FastSAC** uses `SimpleReplayBuffer` (`agents/fast_sac/fast_sac_utils.py`). The buffer allocates `n_env × buf_size × obs_dim × 4B` × 4 tensors on GPU at `algo.setup()` — this is the most common OOM point.
- `agents/callbacks/` contains evaluation callbacks (push, payload, EvalCallbacks).
- `managers/` implements the task manager pattern (commands, observations, rewards, terminations).
- `data/` handles motion file loading for whole-body tracking.

### Inference Pipeline (`holosoma_inference`)

```
run_policy.py
  └── InferenceConfig (Tyro)
        ├── Policy  (locomotion | wbt | dual_mode)  ← policies/
        ├── Input   (joystick | keyboard | ros2)    ← inputs/impl/
        └── SDK     (unitree | booster | mujoco)    ← sdk/
```

Policies are ONNX models. `dual_mode.py` can switch between locomotion and whole-body tracking at runtime.

### Retargeting Pipeline (`holosoma_retargeting`)

```
Motion Capture (C3D / BVH / LAFAN / AMASS / OMOMO / SMPLH / SMPLX)
  └── robot_retarget.py → retargeted NPZ  (demo_results/g1/robot_only/<format>/)
        └── convert_data_format_mj.py → training NPZ (converted_res/robot_only/)
              └── train_agent.py  (via --command...motion_file=)
```

`--line-range start end` in `convert_data_format_mj.py` is the only place frame subsetting is supported.

## Demo Scripts

End-to-end workflow scripts in `demo_scripts/` automate the full pipeline. `workflow_launcher.py` is a PySide6 GUI front-end for these workflows with live training metrics plotting.

The shell scripts use `set -e` and resolve their own path via `BASH_SOURCE` to be runnable from any working directory.
