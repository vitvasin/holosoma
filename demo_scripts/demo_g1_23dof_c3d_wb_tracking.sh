#!/usr/bin/env bash
# demo_scripts/demo_g1_23dof_c3d_wb_tracking.sh
#
# Full pipeline for C3D MoCap data → G1-23DOF whole-body tracking training.
#
# Usage:
#   ./demo_scripts/demo_g1_23dof_c3d_wb_tracking.sh /path/to/motion.c3d [sequence_name]
#
# Arguments:
#   $1  Path to a .c3d file (required)
#   $2  Sequence name override (optional, defaults to c3d filename stem)
#
# Requires Ubuntu/Linux OS (IsaacSim is not supported on Mac)

set -e  # Exit on error

# ── Resolve script location ───────────────────────────────────────────────────
SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── OS check ──────────────────────────────────────────────────────────────────
OS="$(uname -s)"
case "${OS}" in
    Linux*) echo "Detected Linux OS - proceeding..." ;;
    Darwin*) echo "Error: Mac OS is not supported (IsaacSim)."; exit 1 ;;
    *) echo "Error: Unsupported OS: ${OS}."; exit 1 ;;
esac

# ── Arguments ─────────────────────────────────────────────────────────────────
C3D_FILE="${1:-}"
if [ -z "$C3D_FILE" ]; then
    echo "Usage: $0 /path/to/motion.c3d [sequence_name]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/walk1.c3d walk1"
    exit 1
fi
if [ ! -f "$C3D_FILE" ]; then
    echo "Error: C3D file not found: $C3D_FILE"
    exit 1
fi

# Sequence name: use $2 if provided, else strip path/extension from filename
C3D_BASENAME="$(basename "$C3D_FILE" .c3d)"
C3D_BASENAME="${C3D_BASENAME%.C3D}"  # handle uppercase extension too
TASK_NAME="${2:-$C3D_BASENAME}"

echo "C3D file   : $C3D_FILE"
echo "Task name  : $TASK_NAME"

# ── Setup ─────────────────────────────────────────────────────────────────────
echo ""
echo "Sourcing retargeting setup..."
source "$PROJECT_ROOT/scripts/source_retargeting_setup.sh"
pip install -e "$PROJECT_ROOT/src/holosoma_retargeting" --quiet

RETARGET_DIR="$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting"
C3D_DATA_DIR="$RETARGET_DIR/demo_data/c3d"
cd "$RETARGET_DIR"

# ── Step 0: Install ezc3d ─────────────────────────────────────────────────────
echo ""
echo "Checking ezc3d..."
if ! python3 -c "import ezc3d" 2>/dev/null; then
    echo "Installing ezc3d..."
    pip install ezc3d --quiet
fi

# ── Step 1: Convert C3D → NPZ ─────────────────────────────────────────────────
echo ""
echo "Step 1: Converting C3D to NPZ..."
NPZ_OUTPUT="$C3D_DATA_DIR/${TASK_NAME}.npz"
# Optional arguments:
# --marker-set pig (default)  | --marker-set custom
# --marker-map None           | --marker-map /home/smr/holosoma/boxing_markers.json
# --downsample-to 100         | --downsample-to 50
# --lowpass-hz 4.0            | --lowpass-hz 2.0
python3 data_utils/prep_c3d_for_rt.py \
    --input "$C3D_FILE" \
    --output "$NPZ_OUTPUT" \
    --marker-set custom \
    --marker-map /home/smr/holosoma/boxing_markers.json \
    --downsample-to 100 \
    --lowpass-hz 4.0

# ── Step 2: Retargeting ───────────────────────────────────────────────────────
echo ""
RETARGET_OUT="demo_results/g1/robot_only/c3d/${TASK_NAME}.npz"
if [ -f "$RETARGET_OUT" ]; then
    echo "Step 2: Retargeted file already exists:"
    echo "  $RETARGET_OUT"
    printf "  Skip and use existing file? [Y/n]: "
    read -r _reply
    case "${_reply:-Y}" in
        [Nn]*)
            echo "  Overwriting..."
            python examples/robot_retarget.py \
                --robot-config.robot-dof 23 \
                --data_path "$C3D_DATA_DIR" \
                --task-type robot_only \
                --task-name "$TASK_NAME" \
                --data_format c3d \
                --save_dir demo_results/g1/robot_only/c3d
            ;;
        *)
            echo "  Skipping retargeting."
            ;;
    esac
else
    echo "Step 2: Running retargeting..."
    # Optional arguments:
    # --task-type robot_only  |  --task-type object_tracking
    python examples/robot_retarget.py \
        --robot-config.robot-dof 23 \
        --data_path "$C3D_DATA_DIR" \
        --task-type robot_only \
        --task-name "$TASK_NAME" \
        --data_format c3d \
        --save_dir demo_results/g1/robot_only/c3d
fi

# ── Step 3: Data conversion ──────────────────────────────────────────────────
echo ""
CONVERT_OUT="converted_res/robot_only/${TASK_NAME}_mj_fps50.npz"
if [ -f "$CONVERT_OUT" ]; then
    echo "Step 3: Converted file already exists:"
    echo "  $CONVERT_OUT"
    printf "  Skip and use existing file? [Y/n]: "
    read -r _reply
    case "${_reply:-Y}" in
        [Nn]*)
            echo "  Overwriting..."
            python data_conversion/convert_data_format_mj.py \
                --robot-config.robot-dof 23 \
                --input_file "./demo_results/g1/robot_only/c3d/${TASK_NAME}.npz" \
                --output_fps 50 \
                --output_name "$CONVERT_OUT" \
                --data_format c3d \
                --object_name "ground" \
                --line-range 100 500 \
                --once
            ;;
        *)
            echo "  Skipping conversion."
            ;;
    esac
else
    echo "Step 3: Converting to MuJoCo format..."
    # Optional arguments:
    # --object_name "ground"  |  --object_name "box"
    # --line-range 0 250      |  Only process frames 0 to 250 (5 seconds)
    python data_conversion/convert_data_format_mj.py \
        --robot-config.robot-dof 23 \
        --input_file "./demo_results/g1/robot_only/c3d/${TASK_NAME}.npz" \
        --output_fps 50 \
        --output_name "$CONVERT_OUT" \
        --data_format c3d \
        --object_name "ground" \
        --line-range 100 500 \
        --once
fi

# ── Step 4: Source IsaacSim ───────────────────────────────────────────────────
echo ""
echo "Sourcing IsaacSim setup..."
cd "$PROJECT_ROOT"
unset CONDA_ENV_NAME
source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"

HOLOSOMA_DEPS_DIR="${HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}"
pip install -e "$PROJECT_ROOT/src/holosoma[unitree,booster]" --quiet
if ! python -c "import isaaclab" 2>/dev/null; then
    echo "isaaclab not found, reinstalling..."
    pip install 'setuptools<81' --quiet
    echo 'setuptools<81' > /tmp/hs-build-constraints.txt
    PIP_BUILD_CONSTRAINT=/tmp/hs-build-constraints.txt CMAKE_POLICY_VERSION_MINIMUM=3.5 \
        pip install -e "$HOLOSOMA_DEPS_DIR/IsaacLab/source/isaaclab" --quiet
    rm /tmp/hs-build-constraints.txt
fi

# ── Step 5: Training ─────────────────────────────────────────────────────────
echo ""
echo "Step 5: Starting whole-body tracking training..."
# Optional arguments to trial:
# exp:g1-23dof-wbt-fast-sac   | exp:g1-23dof-wbt-fast-sac-no-dr (Disable randomizations)
# --training.headless True    | --training.headless False (Watch the training live)
# --algo.config.alpha-init    | Set to 0.05 for exploration, 0.01 for less
CONVERTED_FILE="$RETARGET_DIR/converted_res/robot_only/${TASK_NAME}_mj_fps50.npz"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/holosoma/holosoma/train_agent.py \
    exp:g1-23dof-wbt-fast-sac \
    logger:wandb-offline \
    --logger.base-dir /media/smr/Data/holosoma_logs \
    --training.headless True \
    --training.num-envs 2048 \
    --algo.config.num-learning-iterations 150000 \
    --algo.config.alpha-init 0.05 \
    --simulator.config.sim.max-episode-length-s 10.0 \
    --logger.video.enabled True \
    --logger.video.interval 10 \
    --observation.groups.actor_obs.history-length 4 \
    --logger.video.save-dir /media/smr/Data/holosoma_logs/videos/g1_23dof_c3d_wbt \
    --command.setup_terms.motion_command.params.motion_config.motion_file=$CONVERTED_FILE

echo ""
echo "Done! Videos saved to: /media/smr/Data/holosoma_logs/videos/g1_23dof_c3d_wbt/"
