#!/usr/bin/env bash

# ───────────────────────────────────────────────────────────────────────────────
# G1-23DOF Whole-Body Tracking Training Script (LAFAN Dataset)
# ───────────────────────────────────────────────────────────────────────────────
# 
# USAGE:
#   ./demo_scripts/demo_g1_23dof_lafan_wb_tracking.sh
#
# This script will:
#   1. Display available LAFAN sequences (from pre-processed .npy files)
#   2. Retarget the selected motion to G1-23DOF robot
#   3. Convert retargeted data to MuJoCo format
#   4. Train whole-body tracking policy
#
# ───────────────────────────────────────────────────────────────────────────────
# ADDING NEW MOTION DATA:
# ───────────────────────────────────────────────────────────────────────────────
# 
# LOCATION TO ADD DATA:
#   $PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting/demo_data/lafan
#
# EXPECTED FILE TYPE:
#   .npy files containing global joint positions with shape (T, J, 3)
#   where T = number of frames, J = number of joints, 3 = [x, y, z]
#
# HOW TO ADD NEW SEQUENCES:
#   1. Prepare your motion data as a .npy file with global joint positions
#   2. Place the .npy file in: src/holosoma_retargeting/holosoma_retargeting/demo_data/lafan/
#   3. Run this script - it will automatically detect and list your new sequence
#   4. Select "new" from the menu to add it for retargeting
#
# NOTE: This script uses LAFAN dataset (.npy format). For other formats (C3D,
# SMPLH, SMPLX, MOCAP), use the corresponding demo script.
# ───────────────────────────────────────────────────────────────────────────────
#
# Requires Ubuntu/Linux OS (IsaacSim is not supported on Mac)

set -e  # Exit on error

# figure out where this file is located even if it is being run from another location
# or as a symlink
SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Detect operating system and check if it's supported
OS="$(uname -s)"
case "${OS}" in
    Linux*)
        MACHINE=Linux
        echo "Detected Linux OS - proceeding..."
        ;;
    Darwin*)
        echo "Error: Mac OS is not supported. This script requires Ubuntu/Linux for IsaacSim."
        exit 1
        ;;
    CYGWIN*|MINGW*)
        echo "Error: Windows is not supported. This script requires Ubuntu/Linux for IsaacSim."
        exit 1
        ;;
    *)
        echo "Error: Unsupported operating system: ${OS}. This script requires Ubuntu/Linux for IsaacSim."
        exit 1
        ;;
esac

# Source retargeting setup script (for retargeting and data conversion)
echo "Sourcing retargeting setup..."
source "$PROJECT_ROOT/scripts/source_retargeting_setup.sh"

# Ensure holosoma_retargeting is installed with correct dependencies (e.g. numpy version)
pip install -e "$PROJECT_ROOT/src/holosoma_retargeting" --quiet

# Change to retargeting directory
RETARGET_DIR="$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting"
cd "$RETARGET_DIR"

# Step 0: Download and process LAFAN data if needed
echo "Checking LAFAN data availability..."
LAFAN_DATA_DIR="$RETARGET_DIR/demo_data/lafan"
LAFAN_TEMP_DIR="$RETARGET_DIR/demo_data/lafan_temp"
LAFAN_ZIP="$RETARGET_DIR/demo_data/lafan1.zip"
DATA_UTILS_DIR="$RETARGET_DIR/data_utils"

# Check if processed LAFAN data already exists
if [ -d "$LAFAN_DATA_DIR" ] && [ "$(ls -A $LAFAN_DATA_DIR/*.npy 2>/dev/null)" ]; then
    echo "LAFAN data already processed."
    echo ""
    echo "=== Available LAFAN Sequences ==="
    echo "Select a sequence to retarget, or add a new one:"
    i=1
    for npy_file in "$LAFAN_DATA_DIR"/*.npy; do
        if [ -f "$npy_file" ]; then
            task_name="$(basename "$npy_file" .npy)"
            echo "  [$i] $task_name"
            i=$((i + 1))
        fi
    done
    echo "  [$i] new  (add new sequence)"
    echo ""
    printf "Enter sequence number (or 'new' to add new sequence): "
    read -r seq_num
    
    if [ "$seq_num" == "new" ] || [ "$seq_num" == "" ]; then
        echo ""
        echo "=== Available LAFAN Sequences ==="
        echo "Select a sequence to retarget, or add a new one:"
        i=1
        for npy_file in "$LAFAN_DATA_DIR"/*.npy; do
            if [ -f "$npy_file" ]; then
                task_name="$(basename "$npy_file" .npy)"
                echo "  [$i] $task_name"
                i=$((i + 1))
            fi
        done
        echo "  [$i] new  (add new sequence)"
        echo ""
        printf "Enter sequence number (or 'new' to add new sequence): "
        read -r seq_num
    fi

    if [ "$seq_num" == "new" ] || [ "$seq_num" == "" ]; then
        echo "Error: No sequence selected. Please run the script and select a valid sequence number."
        exit 1
    fi
    
    TASK_NAME="${seq_num:-1}"
    echo ""
    echo "Selected task: $TASK_NAME"
elif [ ! -d "$LAFAN_DATA_DIR" ]; then
    echo "LAFAN data not found. Downloading and processing..."

    # Create demo_data directory if it doesn't exist
    mkdir -p "$RETARGET_DIR/demo_data"

    # Download lafan1.zip if it doesn't exist
    if [ ! -f "$LAFAN_ZIP" ]; then
        echo "Downloading lafan1.zip..."
        curl -L -o "$LAFAN_ZIP" "https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/master/lafan1/lafan1.zip"
    else
        echo "lafan1.zip already exists. Skipping download."
    fi

    # Uncompress lafan1.zip to temp directory
    if [ ! -d "$LAFAN_TEMP_DIR" ] || [ -z "$(ls -A $LAFAN_TEMP_DIR/*.bvh 2>/dev/null)" ]; then
        echo "Uncompressing lafan1.zip..."
        mkdir -p "$LAFAN_TEMP_DIR"
        unzip -q -o "$LAFAN_ZIP" -d "$LAFAN_TEMP_DIR"
        # Handle different zip structures - move BVH files to top level
        if [ -d "$LAFAN_TEMP_DIR/lafan1/lafan" ]; then
            # Structure: lafan1/lafan/*.bvh
            mv "$LAFAN_TEMP_DIR/lafan1/lafan"/* "$LAFAN_TEMP_DIR/" 2>/dev/null || true
            rm -rf "$LAFAN_TEMP_DIR/lafan1" 2>/dev/null || true
        elif [ -d "$LAFAN_TEMP_DIR/lafan1" ]; then
            # Structure: lafan1/*.bvh
            mv "$LAFAN_TEMP_DIR/lafan1"/* "$LAFAN_TEMP_DIR/" 2>/dev/null || true
            rmdir "$LAFAN_TEMP_DIR/lafan1" 2>/dev/null || true
        fi
    else
        echo "LAFAN BVH files already extracted. Skipping extraction."
    fi

    # Ensure lafan1 processing code is available in data_utils
    if [ ! -d "$DATA_UTILS_DIR/lafan1" ]; then
        echo "Cloning ubisoft-laforge-animation-dataset for processing code..."
        cd "$DATA_UTILS_DIR"
        if [ ! -d "ubisoft-laforge-animation-dataset" ]; then
            git clone -q https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
        fi
        if [ -d "ubisoft-laforge-animation-dataset/lafan1" ] && [ ! -d "lafan1" ]; then
            mv ubisoft-laforge-animation-dataset/lafan1 .
        fi
        cd "$RETARGET_DIR"
    else
        echo "lafan1 processing code already available."
    fi

    # Convert BVH files to .npy format
    echo "Converting BVH files to .npy format..."
    cd "$DATA_UTILS_DIR"
    python extract_global_positions.py --input_dir "$LAFAN_TEMP_DIR" --output_dir "$LAFAN_DATA_DIR"
    cd "$RETARGET_DIR"

    echo ""
    echo "=== Available LAFAN Sequences ==="
    echo "Select a sequence to retarget, or add a new one:"
    i=1
    for npy_file in "$LAFAN_DATA_DIR"/*.npy; do
        if [ -f "$npy_file" ]; then
            task_name="$(basename "$npy_file" .npy)"
            echo "  [$i] $task_name"
            i=$((i + 1))
        fi
    done
    echo "  [$i] new  (add new sequence)"
    echo ""
    printf "Enter sequence number (or 'new' to add new sequence): "
    read -r seq_num

    if [ "$seq_num" == "new" ] || [ "$seq_num" == "" ]; then
        echo ""
        echo "=== Available LAFAN Sequences ==="
        echo "Select a sequence to retarget, or add a new one:"
        i=1
        for npy_file in "$LAFAN_DATA_DIR"/*.npy; do
            if [ -f "$npy_file" ]; then
                task_name="$(basename "$npy_file" .npy)"
                echo "  [$i] $task_name"
                i=$((i + 1))
            fi
        done
        echo "  [$i] new  (add new sequence)"
        echo ""
        printf "Enter sequence number (or 'new' to add new sequence): "
        read -r seq_num
    fi

    if [ "$seq_num" == "new" ] || [ "$seq_num" == "" ]; then
        echo "Error: No sequence selected. Please run the script and select a valid sequence number."
        exit 1
    fi

    TASK_NAME="${seq_num:-1}"
    echo ""
    echo "Selected task: $TASK_NAME"
    echo "LAFAN data processing complete!"
else
    echo "LAFAN data already processed."
    echo ""
    echo "=== Available LAFAN Sequences ==="
    echo "Select a sequence to retarget, or add a new one:"
    i=1
    for npy_file in "$LAFAN_DATA_DIR"/*.npy; do
        if [ -f "$npy_file" ]; then
            task_name="$(basename "$npy_file" .npy)"
            echo "  [$i] $task_name"
            i=$((i + 1))
        fi
    done
    echo "  [$i] new  (add new sequence)"
    echo ""
    printf "Enter sequence number (or 'new' to add new sequence): "
    read -r seq_num

    if [ "$seq_num" == "new" ] || [ "$seq_num" == "" ]; then
        echo ""
        echo "=== Available LAFAN Sequences ==="
        echo "Select a sequence to retarget, or add a new one:"
        i=1
        for npy_file in "$LAFAN_DATA_DIR"/*.npy; do
            if [ -f "$npy_file" ]; then
                task_name="$(basename "$npy_file" .npy)"
                echo "  [$i] $task_name"
                i=$((i + 1))
            fi
        done
        echo "  [$i] new  (add new sequence)"
        echo ""
        printf "Enter sequence number (or 'new' to add new sequence): "
        read -r seq_num
    fi

    if [ "$seq_num" == "new" ] || [ "$seq_num" == "" ]; then
        echo "Error: No sequence selected. Please run the script and select a valid sequence number."
        exit 1
    fi

    TASK_NAME="${seq_num:-1}"
    echo ""
    echo "Selected task: $TASK_NAME"
fi

# Step 1: Run retargeting
RETARGET_OUT="demo_results/g1/robot_only/lafan/$TASK_NAME.npz"
if [ -f "$RETARGET_OUT" ]; then
    echo "Step 1: Retargeted file already exists:"
    echo "  $RETARGET_OUT"
    printf "  Skip and use existing file? [Y/n]: "
    read -r _reply
    case "${_reply:-Y}" in
        [Nn]*)
            echo "  Overwriting..."
            python examples/robot_retarget.py \
                --robot-config.robot-dof 23 \
                --data_path "$LAFAN_DATA_DIR" \
                --task-type robot_only \
                --task-name "$TASK_NAME" \
                --data_format lafan \
                --task-config.ground-range -10 10 \
                --save_dir demo_results/g1/robot_only/lafan \
                --retargeter.foot-sticking-tolerance 0.02
            ;;
        *)
            echo "  Skipping retargeting."
            ;;
    esac
else
    echo "Step 1: Running retargeting..."
    python examples/robot_retarget.py \
        --robot-config.robot-dof 23 \
        --data_path "$LAFAN_DATA_DIR" \
        --task-type robot_only \
        --task-name "$TASK_NAME" \
        --data_format lafan \
        --task-config.ground-range -10 10 \
        --save_dir demo_results/g1/robot_only/lafan \
        --retargeter.foot-sticking-tolerance 0.02
fi

# Step 2: Run data conversion
CONVERT_OUT="converted_res/robot_only/${TASK_NAME}_mj_fps50.npz"
if [ -f "$CONVERT_OUT" ]; then
    echo "Step 2: Converted file already exists:"
    echo "  $CONVERT_OUT"
    printf "  Skip and use existing file? [Y/n]: "
    read -r _reply
    case "${_reply:-Y}" in
        [Nn]*)
            echo "  Overwriting..."
            python data_conversion/convert_data_format_mj.py \
                --robot-config.robot-dof 23 \
                --input_file "./demo_results/g1/robot_only/lafan/${TASK_NAME}.npz" \
                --output_fps 50 \
                --output_name "$CONVERT_OUT" \
                --data_format lafan \
                --object_name "ground" \
                --once
            ;;
        *)
            echo "  Skipping conversion."
            ;;
    esac
else
    echo "Step 2: Converting to MuJoCo format..."
    python data_conversion/convert_data_format_mj.py \
        --robot-config.robot-dof 23 \
        --input_file "./demo_results/g1/robot_only/lafan/${TASK_NAME}.npz" \
        --output_fps 50 \
        --output_name "$CONVERT_OUT" \
        --data_format lafan \
        --object_name "ground" \
        --once
fi

# Step 3: Source IsaacSim setup script (for whole-body tracking training)
echo "Sourcing IsaacSim setup..."
cd "$PROJECT_ROOT"
unset CONDA_ENV_NAME
source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"

# Ensure holosoma and isaaclab are installed in the IsaacSim env
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

# Step 4: Run whole-body tracking training
echo "Running whole-body tracking training..."
CONVERTED_FILE="$RETARGET_DIR/converted_res/robot_only/${TASK_NAME}_mj_fps50.npz"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/holosoma/holosoma/train_agent.py \
    exp:g1-23dof-wbt-fast-sac \
    logger:wandb-offline \
    --training.headless True \
    --training.num-envs 4096 \
    --algo.config.num-learning-iterations 50000 \
    --simulator.config.sim.max-episode-length-s 6.0 \
    --logger.video.enabled True \
    --logger.video.interval 5 \
    --logger.video.save-dir logs/videos/g1_23dof_lafan_wbt \
    --command.setup_terms.motion_command.params.motion_config.motion_file=$CONVERTED_FILE

echo "Done!"
