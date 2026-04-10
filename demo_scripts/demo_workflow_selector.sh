#!/usr/bin/env bash

# ────────────────────────────────────────────────────────────────────────────────
# G1 Robot Whole-Body Tracking Workflow Selector
# ────────────────────────────────────────────────────────────────────────────────
#
# USAGE:
#   ./demo_scripts/demo_workflow_selector.sh
#
# This interactive script provides a menu-driven interface for:
#   1. Selecting retargeting workflows (LAFAN, C3D, etc.)
#   2. Selecting inference workflows (simulation, hardware)
#   3. Browsing available motion files
#   4. Configuring robot type, algorithm, and training options
#
# ────────────────────────────────────────────────────────────────────────────────
# OUTPUT FOLDER CONFIGURATION:
# ────────────────────────────────────────────────────────────────────────────────
#
# Customize these paths in your shell or in a config file:
#   export RETARGET_OUTPUT_DIR="demo_results/g1/robot_only"
#   export CONVERT_OUTPUT_DIR="converted_res"
#   export LOGS_DIR="logs/WholeBodyTracking"
#   export VIDEO_DIR="logs/videos"
#
#   export LAFAN_DATA_DIR="src/holosoma_retargeting/holosoma_retargeting/demo_data/lafan"
#   export C3D_DATA_DIR="src/holosoma_retargeting/holosoma_retargeting/demo_data/c3d"
# ────────────────────────────────────────────────────────────────────────────────
#
# ROBOT TYPES:
#   g1-23dof  - 23 degrees of freedom
#   g1-29dof  - 29 degrees of freedom
#
# ALGORITHMS:
#   g1-23dof-wbt-fast-sac    - Fast SAC (default)
#   g1-23dof-wbt-ppo         - PPO algorithm
#   g1-23dof-wbt-fast-sac-no-dr  - Fast SAC without randomizations
#
# ────────────────────────────────────────────────────────────────────────────────
#

set -e

# ── Resolve script location ─────────────────────────────────────────────────────
SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Output folder configuration (editable) ──────────────────────────────────────
export RETARGET_OUTPUT_DIR="${RETARGET_OUTPUT_DIR:-demo_results/g1/robot_only}"
export CONVERT_OUTPUT_DIR="${CONVERT_OUTPUT_DIR:-converted_res}"
export LOGS_DIR="${LOGS_DIR:-logs/WholeBodyTracking}"
export VIDEO_DIR="${VIDEO_DIR:-logs/videos}"

# ── Data directory configuration ────────────────────────────────────────────────
export LAFAN_DATA_DIR="${LAFAN_DATA_DIR:-$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting/demo_data/lafan}"
export C3D_DATA_DIR="${C3D_DATA_DIR:-$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting/demo_data/c3d}"

# ── Robot type configuration (23dof or 29dof) ──
export ROBOT_DOF="${ROBOT_DOF:-23}"

# ── Algorithm configuration (fast-sac or ppo) ──
export TRAINING_ALGO="${TRAINING_ALGO:-fast-sac}"

# ── Training config defaults ──
export NUM_ENVS="${NUM_ENVS:-4096}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-50000}"
export MAX_EPISODE_LENGTH="${MAX_EPISODE_LENGTH:-6.0}"
export ALPHA_INIT="${ALPHA_INIT:-0.01}"
export GROUND_RANGE="${GROUND_RANGE:--10 10}"

# ── Helper Functions ────────────────────────────────────────────────────────────

show_header() {
  echo ""
  echo "╔══════════════════════════════════════════════════════════════════════════════╗"
  echo "║           G1 Robot Whole-Body Tracking Workflow Selector                       ║"
  echo "╚══════════════════════════════════════════════════════════════════════════════╝"
  echo ""
}

show_main_menu() {
  show_header
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo "  WORKFLOW SELECTION"
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo ""
  echo "  [1] Retargeting + Tracking (LAFAN)"
  echo "      • Select .npy sequences from ${LAFAN_DATA_DIR}"
  echo "      • Retarget to G1-${ROBOT_DOF}DOF robot"
  echo "      • Train whole-body tracking policy with ${TRAINING_ALGO:-fast-sac}"
  echo ""
  echo "  [2] Retargeting + Tracking (C3D)"
  echo "      • Provide a .c3d file path"
  echo "      • Retarget to G1-${ROBOT_DOF}DOF robot"
  echo "      • Train whole-body tracking policy with ${TRAINING_ALGO:-fast-sac}"
  echo ""
  echo "  [3] Retargeting + Tracking (Other Formats)"
  echo "      • SMPLH, SMPLX, MOCAP formats coming soon..."
  echo ""
  echo "  [4] Inference (Simulation)"
  echo "      • Run trained policy in IsaacSim"
  echo "      • Export ONNX model"
  echo ""
  echo "  [5] Inference (Real Robot Hardware)"
  echo "      • Deploy ONNX model to physical G1 robot"
  echo "      • Requires ONNX export first"
  echo ""
  echo "  [6] Browse Available Files"
  echo "      • View LAFAN sequences: $LAFAN_DATA_DIR"
  echo "      • View C3D directory:      $C3D_DATA_DIR"
  echo ""
  echo "  [7] Configure Output Folders"
  echo "      • Customize output paths"
  echo ""
  echo "  [q] Quit"
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo ""
  printf "  Select option [1-7] or [q] to quit: "
}

show_files_menu() {
  show_header
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo "  BROWSE AVAILABLE FILES"
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo ""
  
  echo "  LAFAN Sequences (${LAFAN_DATA_DIR}):"
  if [ -d "$LAFAN_DATA_DIR" ]; then
    count=0
    shopt -s nullglob
    for f in "$LAFAN_DATA_DIR"/*.npy; do
      if [ -f "$f" ]; then
        name="$(basename "$f" .npy)"
        size="$(du -h "$f" 2>/dev/null | cut -f1)"
        echo "    [$count] $name ($size)"
        count=$((count + 1))
      fi
    done
    shopt -u nullglob
    if [ $count -eq 0 ]; then
      echo "    (no .npy files found)"
    fi
    echo "    [$count] new  (add new sequence)"
  else
    echo "    (LAFAN data directory not found - will be created on first use)"
  fi
  echo ""
  
  echo "  C3D Directory (${C3D_DATA_DIR}):"
  if [ -d "$C3D_DATA_DIR" ]; then
    echo "    (place .c3d files here for processing)"
  else
    echo "    (C3D data directory not found)"
  fi
  echo ""
  
  echo "  Checkpoint Files (${LOGS_DIR}):"
  if [ -d "$PROJECT_ROOT/$LOGS_DIR" ]; then
    count=0
    shopt -s nullglob
    for f in "$PROJECT_ROOT/$LOGS_DIR"/*/*model_*.pt; do
      if [ -f "$f" ]; then
        name="$(basename "$f")"
        size="$(du -h "$f" 2>/dev/null | cut -f1)"
        echo "    [$count] $name ($size)"
        count=$((count + 1))
      fi
    done
    shopt -u nullglob
    if [ $count -eq 0 ]; then
      echo "    (no checkpoint files found)"
    fi
    echo "    [$count] latest  (use latest checkpoint)"
  else
    echo "    (logs directory not found)"
  fi
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo ""
  printf "  Select option or [b] back: "
}

show_config_menu() {
  show_header
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo "  OUTPUT FOLDER CONFIGURATION"
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo ""
  echo "  Current Configuration:"
  echo "    Retarget Output:  $RETARGET_OUTPUT_DIR"
  echo "    Convert Output:   $CONVERT_OUTPUT_DIR"
  echo "    Logs Directory:   $LOGS_DIR"
  echo "    Videos Directory: $VIDEO_DIR"
  echo ""
  echo "  LAFAN Data Dir:     $LAFAN_DATA_DIR"
  echo "  C3D Data Dir:       $C3D_DATA_DIR"
  echo ""
  echo "  EDITING CONFIGURATION:"
  echo "    1) Copy this script to your home directory"
  echo "    2) Edit variables at the top (lines 34-43)"
  echo "    3) Or create ~/.holosoma_workflow_config.sh and source it"
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo ""
  printf "  Select option [1-2] or [b] back: "
}

# Retarget LAFAN workflow
workflow_retarget_lafan() {
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo "  WORKFLOW: LAFAN Retargeting + Tracking"
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo ""
  
  echo "  Robot Type:     G1-${ROBOT_DOF}DOF"
  echo "  Algorithm:      ${TRAINING_ALGO:-fast-sac}"
  echo ""
  
  # List available sequences
  echo "  Available LAFAN sequences:"
  i=1
  LAFAN_TASKS=()
  if [ -d "$LAFAN_DATA_DIR" ]; then
    shopt -s nullglob
    for npy_file in "$LAFAN_DATA_DIR"/*.npy; do
      if [ -f "$npy_file" ]; then
        task_name="$(basename "$npy_file" .npy)"
        echo "    [$i] $task_name"
        LAFAN_TASKS+=("$task_name")
        i=$((i + 1))
      fi
    done
    shopt -u nullglob
  fi
  echo "    [$i] new  (add new sequence)"
  echo ""
  printf "  Enter sequence number (or 'new'): "
  read -r seq_input
  
  if [ "$seq_input" = "new" ] || [ -z "$seq_input" ]; then
    echo "  Error: Please select a valid sequence or provide a path."
    echo ""
    printf "  Go back [b] or retry [r]: "
    read -r go_back
    if [ "$go_back" = "b" ] || [ "$go_back" = "B" ]; then
      return
    fi
    return
  fi
  
  # Resolve sequence number to task name
  if [[ "$seq_input" =~ ^[0-9]+$ ]]; then
    index=$((seq_input - 1))
    if [ $index -ge 0 ] && [ $index -lt ${#LAFAN_TASKS[@]} ]; then
      TASK_NAME="${LAFAN_TASKS[$index]}"
    else
      echo "  Error: Invalid sequence number."
      return
    fi
  else
    TASK_NAME="$seq_input"
  fi
  if [ -z "$TASK_NAME" ]; then
    echo "  Error: No sequence selected."
    return
  fi
  
  echo ""
  echo "  Selected task: $TASK_NAME"
  
  # Display output folder info
  echo ""
  echo "  Output folders:"
  echo "    Retarget:  $RETARGET_OUTPUT_DIR/$TASK_NAME.npz"
  echo "    Convert:   $CONVERT_OUTPUT_DIR/${TASK_NAME}_mj_fps50.npz"
  echo ""
  echo "  Running retargeting pipeline..."
  echo ""
  
  # Source retargeting setup
  echo "  Sourcing retargeting setup..."
  source "$PROJECT_ROOT/scripts/source_retargeting_setup.sh"
  pip install -e "$PROJECT_ROOT/src/holosoma_retargeting" --quiet
  
  RETARGET_DIR="$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting"
  cd "$RETARGET_DIR"
  
  # Check if data needs to be downloaded/processed
  LAFAN_DATA_DIR="$RETARGET_DIR/demo_data/lafan"
  
  if [ -d "$LAFAN_DATA_DIR" ] && [ "$(ls -A $LAFAN_DATA_DIR/*.npy 2>/dev/null)" ]; then
    echo "  LAFAN data already processed."
  else
    echo "  LAFAN data not found. Downloading and processing..."
    
    mkdir -p "$RETARGET_DIR/demo_data"
    
    LAFAN_ZIP="$RETARGET_DIR/demo_data/lafan1.zip"
    if [ ! -f "$LAFAN_ZIP" ]; then
      echo "  Downloading lafan1.zip..."
      curl -L -o "$LAFAN_ZIP" "https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/master/lafan1/lafan1.zip"
    fi
    
    LAFAN_TEMP_DIR="$RETARGET_DIR/demo_data/lafan_temp"
    if [ ! -d "$LAFAN_TEMP_DIR" ] || [ -z "$(ls -A $LAFAN_TEMP_DIR/*.bvh 2>/dev/null)" ]; then
      mkdir -p "$LAFAN_TEMP_DIR"
      unzip -q -o "$LAFAN_ZIP" -d "$LAFAN_TEMP_DIR"
      if [ -d "$LAFAN_TEMP_DIR/lafan1/lafan" ]; then
        mv "$LAFAN_TEMP_DIR/lafan1/lafan"/* "$LAFAN_TEMP_DIR/" 2>/dev/null || true
        rm -rf "$LAFAN_TEMP_DIR/lafan1" 2>/dev/null || true
      elif [ -d "$LAFAN_TEMP_DIR/lafan1" ]; then
        mv "$LAFAN_TEMP_DIR/lafan1"/* "$LAFAN_TEMP_DIR/" 2>/dev/null || true
        rmdir "$LAFAN_TEMP_DIR/lafan1" 2>/dev/null || true
      fi
    fi
    
    DATA_UTILS_DIR="$RETARGET_DIR/data_utils"
    if [ ! -d "$DATA_UTILS_DIR/lafan1" ]; then
      cd "$DATA_UTILS_DIR"
      if [ ! -d "ubisoft-laforge-animation-dataset" ]; then
        git clone -q https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
      fi
      if [ -d "ubisoft-laforge-animation-dataset/lafan1" ] && [ ! -d "lafan1" ]; then
        mv ubisoft-laforge-animation-dataset/lafan1 .
      fi
      cd "$RETARGET_DIR"
    fi
    
    echo "  Converting BVH files to .npy format..."
    cd "$DATA_UTILS_DIR"
    python extract_global_positions.py --input_dir "$LAFAN_TEMP_DIR" --output_dir "$LAFAN_DATA_DIR"
    cd "$RETARGET_DIR"
    
    echo "  LAFAN data processing complete!"
  fi
  
  # Step 1: Retargeting
  RETARGET_OUT="$RETARGET_OUTPUT_DIR/$TASK_NAME.npz"
  if [ -f "$RETARGET_OUT" ]; then
    echo ""
    echo "  Retargeted file already exists:"
    echo "    $RETARGET_OUT"
    printf "  Skip and use existing file? [Y/n]: "
    read -r _reply
    case "${_reply:-Y}" in
      [Nn]*)
        echo "  Overwriting..."
        python examples/robot_retarget.py \
          --robot-config.robot-dof "$ROBOT_DOF" \
          --data_path "$LAFAN_DATA_DIR" \
          --task-type robot_only \
          --task-name "$TASK_NAME" \
          --data_format lafan \
          --task-config.ground-range $GROUND_RANGE \
          --save_dir "$RETARGET_OUTPUT_DIR" \
          --retargeter.foot-sticking-tolerance 0.02
        ;;
      *)
        echo "  Skipping retargeting."
        ;;
    esac
  else
    echo "  Running retargeting..."
    python examples/robot_retarget.py \
      --robot-config.robot-dof "$ROBOT_DOF" \
      --data_path "$LAFAN_DATA_DIR" \
      --task-type robot_only \
      --task-name "$TASK_NAME" \
      --data_format lafan \
      --task-config.ground-range $GROUND_RANGE \
      --save_dir "$RETARGET_OUTPUT_DIR" \
      --retargeter.foot-sticking-tolerance 0.02
  fi
  
  # Step 2: Data conversion
  CONVERT_OUT="$CONVERT_OUTPUT_DIR/${TASK_NAME}_mj_fps50.npz"
  if [ -f "$CONVERT_OUT" ]; then
    echo ""
    echo "  Converted file already exists:"
    echo "    $CONVERT_OUT"
    printf "  Skip and use existing file? [Y/n]: "
    read -r _reply
    case "${_reply:-Y}" in
      [Nn]*)
        echo "  Overwriting..."
        python data_conversion/convert_data_format_mj.py \
          --robot-config.robot-dof "$ROBOT_DOF" \
          --input_file "$RETARGET_OUTPUT_DIR/$TASK_NAME.npz" \
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
    echo "  Converting to MuJoCo format..."
    python data_conversion/convert_data_format_mj.py \
      --robot-config.robot-dof "$ROBOT_DOF" \
      --input_file "$RETARGET_OUTPUT_DIR/$TASK_NAME.npz" \
      --output_fps 50 \
      --output_name "$CONVERT_OUT" \
      --data_format lafan \
      --object_name "ground" \
      --once
  fi
  
  # Step 3: Source IsaacSim
  echo ""
  echo "  Sourcing IsaacSim setup..."
  cd "$PROJECT_ROOT"
  unset CONDA_ENV_NAME
  source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"
  
  HOLOSOMA_DEPS_DIR="${HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}"
  pip install -e "$PROJECT_ROOT/src/holosoma[unitree,booster]" --quiet
  if ! python -c "import isaaclab" 2>/dev/null; then
    echo "  isaaclab not found, reinstalling..."
    pip install 'setuptools<81' --quiet
    echo 'setuptools<81' > /tmp/hs-build-constraints.txt
    PIP_BUILD_CONSTRAINT=/tmp/hs-build-constraints.txt CMAKE_POLICY_VERSION_MINIMUM=3.5 \
      pip install -e "$HOLOSOMA_DEPS_DIR/IsaacLab/source/isaaclab" --quiet
    rm /tmp/hs-build-constraints.txt
  fi
  
  # Step 4: Training
  echo ""
  echo "  Running whole-body tracking training..."
  CONVERTED_FILE="$RETARGET_DIR/$CONVERT_OUTPUT_DIR/${TASK_NAME}_mj_fps50.npz"
  
  # Build experiment name based on robot DOF and algorithm
  ROBOT_STEM="23dof"
  if [ "$ROBOT_DOF" = "29" ]; then
    ROBOT_STEM="29dof"
  fi
  
  EXP_NAME="g1-${ROBOT_STEM}-wbt-${TRAINING_ALGO}"
  
  case "$TRAINING_ALGO" in
    ppo)
      ALGO_ARGS="--algo.config.num-learning-iterations $NUM_ITERATIONS"
      ;;
    *)
      ALGO_ARGS="--algo.config.num-learning-iterations $NUM_ITERATIONS"
      ;;
  esac
  
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/holosoma/holosoma/train_agent.py \
    exp:${EXP_NAME} \
    logger:wandb-offline \
    --training.headless True \
    --training.num-envs $NUM_ENVS \
    ${ALGO_ARGS} \
    --simulator.config.sim.max-episode-length-s $MAX_EPISODE_LENGTH \
    --logger.video.enabled True \
    --logger.video.interval 5 \
    --logger.video.save-dir "$VIDEO_DIR/g1_${ROBOT_STEM}_lafan_wbt" \
    --command.setup_terms.motion_command.params.motion_config.motion_file=$CONVERTED_FILE
  
  echo ""
  echo "  Training complete!"
  echo "  Videos saved to: $VIDEO_DIR/g1_${ROBOT_STEM}_lafan_wbt/"
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════════════"
}

# Retarget C3D workflow
workflow_retarget_c3d() {
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo "  WORKFLOW: C3D Retargeting + Tracking"
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo ""
  
  echo "  Robot Type:     G1-${ROBOT_DOF}DOF"
  echo "  Algorithm:      ${TRAINING_ALGO:-fast-sac}"
  echo ""
  
  echo "  C3D Workflow requires a .c3d file path."
  echo "  Place your .c3d file in: $C3D_DATA_DIR"
  echo "  Or provide the full path to a .c3d file below."
  echo ""
  echo "  Available sequences in: $C3D_DATA_DIR"
  echo ""
  
  # List available C3D files if any
  if [ -d "$C3D_DATA_DIR" ]; then
    count=0
    shopt -s nullglob
    for f in "$C3D_DATA_DIR"/*.c3d; do
      if [ -f "$f" ]; then
        name="$(basename "$f")"
        echo "    [$count] $name"
        count=$((count + 1))
      fi
    done
    shopt -u nullglob
    if [ $count -eq 0 ]; then
      echo "    (no .c3d files found)"
    fi
  fi
  echo ""
  echo "  Enter C3D file path (or file number above, or 'new' for new file):"
  read -r c3d_input
  
  if [ "$c3d_input" = "new" ] || [ -z "$c3d_input" ]; then
    echo ""
    echo "  To use a new .c3d file:"
    echo "    1. Place it in: $C3D_DATA_DIR"
    echo "    2. Run this script again"
    echo ""
    printf "  Go back [b]: "
    read -r go_back
    if [ "$go_back" != "b" ]; then
      printf "  Please enter a valid file path: "
      read -r c3d_input
    fi
  fi
  
  if [ -z "$c3d_input" ]; then
    echo "  Error: No C3D file path provided."
    return
  fi
  
  # Extract task name from file path or use filename
  C3D_BASENAME="$(basename "$c3d_input" .c3d)"
  C3D_BASENAME="${C3D_BASENAME%.C3D}"
  TASK_NAME="${C3D_BASENAME}"
  
  echo ""
  echo "  C3D file   : $c3d_input"
  echo "  Task name  : $TASK_NAME"
  
  echo ""
  echo "  Output folders:"
  echo "    Retarget:  $RETARGET_OUTPUT_DIR/c3d/$TASK_NAME.npz"
  echo "    Convert:   $CONVERT_OUTPUT_DIR/${TASK_NAME}_mj_fps50.npz"
  echo ""
  echo "  Running C3D retargeting pipeline..."
  echo ""
  
  # Source retargeting setup
  echo "  Sourcing retargeting setup..."
  source "$PROJECT_ROOT/scripts/source_retargeting_setup.sh"
  pip install -e "$PROJECT_ROOT/src/holosoma_retargeting" --quiet
  
  RETARGET_DIR="$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting"
  C3D_DATA_DIR="$RETARGET_DIR/demo_data/c3d"
  cd "$RETARGET_DIR"
  
  # Install ezc3d
  echo ""
  echo "  Checking ezc3d..."
  if ! python3 -c "import ezc3d" 2>/dev/null; then
    echo "  Installing ezc3d..."
    pip install ezc3d --quiet
  fi
  
  # Step 1: Convert C3D → NPZ
  echo ""
  echo "  Step 1: Converting C3D to NPZ..."
  NPZ_OUTPUT="$C3D_DATA_DIR/${TASK_NAME}.npz"
  python3 data_utils/prep_c3d_for_rt.py \
    --input "$c3d_input" \
    --output "$NPZ_OUTPUT" \
    --marker-set custom \
    --marker-map "$PROJECT_ROOT/boxing_markers.json" \
    --downsample-to 100 \
    --lowpass-hz 4.0
  
  # Step 2: Retargeting
  echo ""
  echo "  Step 2: Retargeting..."
  RETARGET_OUT="$RETARGET_OUTPUT_DIR/c3d/${TASK_NAME}.npz"
  if [ -f "$RETARGET_OUT" ]; then
    echo "  Retargeted file already exists:"
    echo "    $RETARGET_OUT"
    printf "  Skip and use existing file? [Y/n]: "
    read -r _reply
    case "${_reply:-Y}" in
      [Nn]*)
        echo "  Overwriting..."
        python examples/robot_retarget.py \
          --robot-config.robot-dof "$ROBOT_DOF" \
          --data_path "$C3D_DATA_DIR" \
          --task-type robot_only \
          --task-name "$TASK_NAME" \
          --data_format c3d \
          --save_dir "$RETARGET_OUTPUT_DIR/c3d"
        ;;
      *)
        echo "  Skipping retargeting."
        ;;
    esac
  else
    echo "  Running retargeting..."
    python examples/robot_retarget.py \
      --robot-config.robot-dof "$ROBOT_DOF" \
      --data_path "$C3D_DATA_DIR" \
      --task-type robot_only \
      --task-name "$TASK_NAME" \
      --data_format c3d \
      --save_dir "$RETARGET_OUTPUT_DIR/c3d"
  fi
  
  # Step 3: Data conversion
  echo ""
  echo "  Step 3: Data conversion..."
  CONVERT_OUT="$CONVERT_OUTPUT_DIR/${TASK_NAME}_mj_fps50.npz"
  if [ -f "$CONVERT_OUT" ]; then
    echo "  Converted file already exists:"
    echo "    $CONVERT_OUT"
    printf "  Skip and use existing file? [Y/n]: "
    read -r _reply
    case "${_reply:-Y}" in
      [Nn]*)
        echo "  Overwriting..."
        python data_conversion/convert_data_format_mj.py \
          --robot-config.robot-dof "$ROBOT_DOF" \
          --input_file "$RETARGET_OUTPUT_DIR/c3d/${TASK_NAME}.npz" \
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
    echo "  Converting to MuJoCo format..."
    python data_conversion/convert_data_format_mj.py \
      --robot-config.robot-dof "$ROBOT_DOF" \
      --input_file "$RETARGET_OUTPUT_DIR/c3d/${TASK_NAME}.npz" \
      --output_fps 50 \
      --output_name "$CONVERT_OUT" \
      --data_format c3d \
      --object_name "ground" \
      --line-range 100 500 \
      --once
  fi
  
  # Step 4: IsaacSim setup
  echo ""
  echo "  Sourcing IsaacSim setup..."
  cd "$PROJECT_ROOT"
  unset CONDA_ENV_NAME
  source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"
  
  HOLOSOMA_DEPS_DIR="${HOLOSOMA_DEPS_DIR:-$HOME/.holosoma_deps}"
  pip install -e "$PROJECT_ROOT/src/holosoma[unitree,booster]" --quiet
  if ! python -c "import isaaclab" 2>/dev/null; then
    echo "  isaaclab not found, reinstalling..."
    pip install 'setuptools<81' --quiet
    echo 'setuptools<81' > /tmp/hs-build-constraints.txt
    PIP_BUILD_CONSTRAINT=/tmp/hs-build-constraints.txt CMAKE_POLICY_VERSION_MINIMUM=3.5 \
      pip install -e "$HOLOSOMA_DEPS_DIR/IsaacLab/source/isaaclab" --quiet
    rm /tmp/hs-build-constraints.txt
  fi
  
  # Step 5: Training
  echo ""
  echo "  Step 5: Starting whole-body tracking training..."
  CONVERTED_FILE="$RETARGET_DIR/$CONVERT_OUTPUT_DIR/${TASK_NAME}_mj_fps50.npz"
  
  # Build experiment name based on robot DOF and algorithm
  ROBOT_STEM="23dof"
  if [ "$ROBOT_DOF" = "29" ]; then
    ROBOT_STEM="29dof"
  fi
  
  EXP_NAME="g1-${ROBOT_STEM}-wbt-${TRAINING_ALGO}"
  
  case "$TRAINING_ALGO" in
    ppo)
      ALGO_ARGS="--algo.config.num-learning-iterations $NUM_ITERATIONS"
      ;;
    fast-sac|fast-sac-no-dr)
      if [ "$TRAINING_ALGO" = "fast-sac-no-dr" ]; then
        EXP_NAME="g1-${ROBOT_STEM}-wbt-${TRAINING_ALGO#-}"
      fi
      ALGO_ARGS="--algo.config.num-learning-iterations $NUM_ITERATIONS"
      ;;
    *)
      ALGO_ARGS="--algo.config.num-learning-iterations $NUM_ITERATIONS"
      ;;
  esac
  
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/holosoma/holosoma/train_agent.py \
    exp:${EXP_NAME} \
    logger:wandb-offline \
    --logger.base-dir "$LOGS_DIR" \
    --training.headless True \
    --training.num-envs 2048 \
    ${ALGO_ARGS} \
    --simulator.config.sim.max-episode-length-s 10.0 \
    --logger.video.enabled True \
    --logger.video.interval 10 \
    --observation.groups.actor_obs.history-length 4 \
    --logger.video.save-dir "$VIDEO_DIR/g1_${ROBOT_STEM}_c3d_wbt" \
    --command.setup_terms.motion_command.params.motion_config.motion_file=$CONVERTED_FILE
  
  echo ""
  echo "  Training complete!"
  echo "  Videos saved to: $VIDEO_DIR/g1_${ROBOT_STEM}_c3d_wbt/"
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════════════"
}

# Inference simulation
workflow_inference_sim() {
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo "  INFERENCE MODE: Simulation"
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo ""
  echo "  Simulation inference requires a trained checkpoint."
  echo ""
  echo "  Available checkpoints in: $LOGS_DIR"
  echo ""
  
  # List available checkpoints
  if [ -d "$PROJECT_ROOT/$LOGS_DIR" ]; then
    echo "  Checkpoints:"
    count=0
    shopt -s nullglob
    for f in "$PROJECT_ROOT/$LOGS_DIR"/*/*model_*.pt; do
      if [ -f "$f" ]; then
        name="$(basename "$f")"
        echo "    [$count] $name"
        count=$((count + 1))
      fi
    done
    shopt -u nullglob
    if [ $count -eq 0 ]; then
      echo "    (no checkpoints found)"
      echo ""
      echo "  To train a checkpoint, use workflow [1] or [2] first."
      echo ""
      printf "  Go back [b]: "
      read -r go_back
      return
    fi
  else
    echo "  (logs directory not found)"
  fi
  echo ""
  
  echo "  Enter checkpoint file path or number above:"
  read -r checkpoint_input
  
  # Resolve checkpoint path
  if [[ "$checkpoint_input" =~ ^[0-9]+$ ]]; then
    # User entered a number
    if [ -d "$PROJECT_ROOT/$LOGS_DIR" ]; then
      idx=0
      shopt -s nullglob
      for f in "$PROJECT_ROOT/$LOGS_DIR"/*/*model_*.pt; do
        if [ -f "$f" ]; then
          if [ $idx -eq "$checkpoint_input" ]; then
            CHECKPOINT="$f"
            break
          fi
          idx=$((idx + 1))
        fi
      done
      shopt -u nullglob
    fi
  else
    # User entered a path
    CHECKPOINT="$checkpoint_input"
  fi
  
  if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "  Error: Checkpoint not found."
    echo ""
    printf "  Go back [b]: "
    read -r go_back
    return
  fi
  
  echo ""
  echo "  Checkpoint: $CHECKPOINT"
  echo ""
  echo "  Running simulation inference..."
  echo "  Headless mode: true (no GUI)"
  echo ""
  
  # Setup IsaacSim env
  echo "  Sourcing IsaacSim env..."
  unset CONDA_ENV_NAME
  source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"
  pip install -e "$PROJECT_ROOT/src/holosoma[unitree,booster]" --quiet
  
  # Checkpoint directory
  CHECKPOINT_DIR="$(dirname "$CHECKPOINT")"
  CHECKPOINT_STEM="$(basename "$CHECKPOINT" .pt)"
  ONNX_EXPORT_PATH="$CHECKPOINT_DIR/exported/${CHECKPOINT_STEM}.onnx"
  mkdir -p "$(dirname "$ONNX_EXPORT_PATH")"
  
  # Run evaluation
  cd "$PROJECT_ROOT"
  echo "  Launching IsaacSim evaluation..."
  echo "  (Press Ctrl+C to stop)"
  echo ""
  
  python src/holosoma/holosoma/eval_agent.py \
    --checkpoint "$CHECKPOINT" \
    --training.export-onnx True \
    --training.num-envs 1 \
    --training.headless True
  
  echo ""
  if [ -f "$ONNX_EXPORT_PATH" ]; then
    echo "  ONNX exported to: $ONNX_EXPORT_PATH"
    echo ""
    echo "  To deploy on real robot:"
    echo "    Run workflow [5] with: $ONNX_EXPORT_PATH"
  fi
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════════════"
}

# Inference hardware
workflow_inference_hardware() {
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo "  INFERENCE MODE: Real Robot Hardware"
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo ""
  echo "⚠️  WARNING: This will control the physical G1 robot!"
  echo "   Make sure the robot is in a safe position and E-stop is accessible."
  echo ""
  echo "  Hardware inference requires an ONNX model file."
  echo ""
  
  # List available ONNX files
  echo "  Available ONNX models:"
  onnx_count=0
  onnx_list=""
  
  # Search for ONNX files in common locations
  shopt -s nullglob
  for dir in "$PROJECT_ROOT/logs" "$PROJECT_ROOT/converted_res"; do
    if [ -d "$dir" ]; then
      for f in "$dir"/*.onnx; do
        if [ -f "$f" ]; then
          echo "    [$onnx_count] $(basename "$f")"
          echo "     (path: $f)"
          onnx_count=$((onnx_count + 1))
          onnx_list="$onnx_list|$f"
        fi
      done
    fi
  done
  shopt -u nullglob
  
  if [ $onnx_count -eq 0 ]; then
    echo "    (no ONNX files found)"
    echo ""
    echo "  To export ONNX from a trained checkpoint:"
    echo "    Run workflow [4] (Inference - Simulation) first"
    echo ""
    printf "  Go back [b]: "
    read -r go_back
    return
  fi
  
  echo ""
  echo "  Enter ONNX file number above:"
  read -r onnx_input
  
  # Resolve ONNX path
  if [[ "$onnx_input" =~ ^[0-9]+$ ]]; then
    IFS='|' read -ra onnx_arr <<< "$onnx_list"
    onnx_idx=0
    for path in "${onnx_arr[@]}"; do
      if [ -z "$path" ]; then continue; fi
      if [ "$onnx_idx" -eq "$onnx_input" ]; then
        ONNX_PATH="$path"
        break
      fi
      onnx_idx=$((onnx_idx + 1))
    done
  else
    ONNX_PATH="$onnx_input"
  fi
  
  if [ -z "$ONNX_PATH" ] || [ ! -f "${ONNX_PATH:-}" ]; then
    echo "  Error: ONNX file not found."
    echo ""
    printf "  Go back [b]: "
    read -r go_back
    return
  fi
  
  echo ""
  echo "  ONNX file: $ONNX_PATH"
  echo ""
  echo "  Proceeding to hardware deployment..."
  echo ""
  
  # Safety confirmation
  printf "   Continue? [y/N]: "
  read -r _confirm
  case "${_confirm}" in
    [Yy]*) echo "Proceeding..." ;;
    *) echo "Aborted."; return 0 ;;
  esac
  
  # Setup inference env
  echo ""
  echo "  Sourcing inference env (hsinference)..."
  unset CONDA_ENV_NAME
  source "$PROJECT_ROOT/scripts/source_inference_setup.sh"
  pip install -e "$PROJECT_ROOT/src/holosoma_inference" --quiet
  
  echo ""
  echo "  Keyboard controls (keep this terminal focused):"
  echo "    ]   Start policy"
  echo "    o   Stop policy"
  echo "    i   Default pose"
  echo "    s   Start motion clip"
  echo ""
  
  cd "$PROJECT_ROOT"
  python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-23dof-wbt \
    --task.model-path="$ONNX_PATH" \
    --observation.groups.actor_obs.history-length=4
  
  echo ""
  echo "  Inference complete."
  echo "═══════════════════════════════════════════════════════════════════════════════"
}

# Browse files
browse_files() {
  show_files_menu
  echo ""
  
  while true; do
    read -r choice
    
    case "$choice" in
      0)
        echo "  Listing LAFAN sequences..."
        if [ -d "$LAFAN_DATA_DIR" ]; then
          shopt -s nullglob
          i=1
          for f in "$LAFAN_DATA_DIR"/*.npy; do
            if [ -f "$f" ]; then
              name="$(basename "$f" .npy)"
              echo "    [$i] $name"
              i=$((i + 1))
            fi
          done
          shopt -u nullglob
        fi
        echo "    [$i] new"
        ;;
      1)
        echo "  Listing C3D files..."
        if [ -d "$C3D_DATA_DIR" ]; then
          shopt -s nullglob
          i=0
          for f in "$C3D_DATA_DIR"/*.c3d; do
            if [ -f "$f" ]; then
              name="$(basename "$f")"
              echo "    [$i] $name"
              i=$((i + 1))
            fi
          done
          shopt -u nullglob
        fi
        ;;
      2)
        echo "  Listing checkpoints..."
        if [ -d "$PROJECT_ROOT/$LOGS_DIR" ]; then
          shopt -s nullglob
          i=0
          for f in "$PROJECT_ROOT/$LOGS_DIR"/*/*model_*.pt; do
            if [ -f "$f" ]; then
              name="$(basename "$f")"
              echo "    [$i] $name"
              i=$((i + 1))
            fi
          done
          shopt -u nullglob
        fi
        ;;
      b|B) return 0 ;;
      q|Q|exit|Exit) exit 0 ;;
      *) echo "Invalid option. Press 'b' to go back." ;;
    esac
  done
}

# Configure output folders
configure_output() {
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo "  OUTPUT FOLDER CONFIGURATION"
  echo "═══════════════════════════════════════════════════════════════════════════════"
  echo ""
  echo "  Current Configuration:"
  echo "    Retarget Output:  $RETARGET_OUTPUT_DIR"
  echo "    Convert Output:   $CONVERT_OUTPUT_DIR"
  echo "    Logs Directory:   $LOGS_DIR"
  echo "    Videos Directory: $VIDEO_DIR"
  echo ""
  echo "  LAFAN_DATA_DIR:         $LAFAN_DATA_DIR"
  echo "  C3D_DATA_DIR:           $C3D_DATA_DIR"
  echo ""
  echo "  Robot Type:             G1-${ROBOT_DOF}DOF"
  echo "  Training Algorithm:     ${TRAINING_ALGO:-fast-sac}"
  echo ""
  echo "  Training Options:"
  echo "    NUM_ENVS:             $NUM_ENVS"
  echo "    NUM_ITERATIONS:       $NUM_ITERATIONS"
  echo "    MAX_EPISODE_LENGTH:   $MAX_EPISODE_LENGTH"
  echo "    ALPHA_INIT:           $ALPHA_INIT"
  echo ""
  echo "  Instructions:"
  echo "    1. Copy this script to your home directory"
  echo "    2. Edit variables at the top (lines 34-43)"
  echo "    3. Or create ~/.holosoma_workflow_config.sh and source it"
  echo ""
  echo "  Example custom config:"
  echo "    # ~/.holosoma_workflow_config.sh"
  echo "    export ROBOT_DOF=29"
  echo "    export TRAINING_ALGO=ppo"
  echo "    export NUM_ENVS=2048"
  echo "    export NUM_ITERATIONS=100000"
  echo ""
  
  echo "═══════════════════════════════════════════════════════════════════════════════"
}

# ── Main Loop ───────────────────────────────────────────────────────────────────
while true; do
  # Show menu and get input
  show_main_menu
  read -r selection
  
  case "$selection" in
    1)
      workflow_retarget_lafan
      ;;
    2)
      workflow_retarget_c3d
      ;;
    3)
      echo ""
      echo "  Other data formats (SMPLH, SMPLX, MOCAP) are not yet supported."
      echo "  Use the corresponding demo script:"
      echo "    demo_g1_23dof_wb_tracking.sh   (generic format)"
      echo "    demo_lafan_wb_tracking.sh      (LAFAN)"
      echo "    demo_c3d_wb_tracking.sh        (C3D)"
      echo ""
      printf "  Go back [b]: "
      read -r go_back
      ;;
    4)
      workflow_inference_sim
      ;;
    5)
      workflow_inference_hardware
      ;;
    6)
      browse_files
      ;;
    7)
      configure_output
      ;;
    q|Q|exit|Exit)
      echo ""
      echo "  Goodbye!"
      exit 0
      ;;
    *)
      echo "Invalid option. Please try again."
      ;;
  esac
  
  # Press enter for next iteration
  echo ""
  printf "  Press Enter to continue... "
  read -r _dummy
done