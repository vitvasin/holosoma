#!/usr/bin/env bash
# demo_scripts/demo_g1_23dof_inference.sh
#
# Evaluate a trained G1-23DOF WBT policy.
#
# MODES:
#   --sim       Run in IsaacSim (default, no hardware needed)
#   --hardware  Deploy on real G1 robot (requires ONNX + physical robot)
#
# Usage:
#   # Simulation (IsaacSim, default):
#   ./demo_scripts/demo_g1_23dof_inference.sh --checkpoint logs/WholeBodyTracking/.../model_50000.pt
#   ./demo_scripts/demo_g1_23dof_inference.sh --checkpoint logs/WholeBodyTracking/.../model_50000.pt --sim
#
#   # Real robot (requires ONNX export first):
#   ./demo_scripts/demo_g1_23dof_inference.sh --onnx path/to/model.onnx --hardware

set -e

# ── Resolve script location ───────────────────────────────────────────────────
SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Parse arguments ───────────────────────────────────────────────────────────
CHECKPOINT=""
ONNX_PATH=""
MODE="sim"        # default: simulation
HEADLESS="False"  # show IsaacSim window by default
NUM_ENVS=1

while [ $# -gt 0 ]; do
    case "$1" in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --onnx)       ONNX_PATH="$2";  shift 2 ;;
        --sim)        MODE="sim";       shift ;;
        --hardware)   MODE="hardware";  shift ;;
        --headless)   HEADLESS="True";  shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint <path>   .pt checkpoint file (for --sim mode)"
            echo "  --onnx <path>         .onnx model file (for --hardware mode)"
            echo "  --sim                 Run in IsaacSim simulation (default)"
            echo "  --hardware            Deploy on real G1-23DOF robot"
            echo "  --headless            IsaacSim headless mode (no GUI)"
            echo ""
            echo "Simulation example:"
            echo "  $0 --checkpoint logs/WholeBodyTracking/.../model_0050000.pt"
            echo ""
            echo "Hardware example:"
            echo "  $0 --onnx logs/WholeBodyTracking/.../exported/model_0050000.onnx --hardware"
            echo ""
            echo "Latest checkpoints:"
            ls -t "$PROJECT_ROOT"/logs/WholeBodyTracking/*/model_*.pt 2>/dev/null | head -5 || echo "  (none found)"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate arguments ────────────────────────────────────────────────────────
if [ "$MODE" = "sim" ] && [ -z "$CHECKPOINT" ]; then
    echo "Error: --sim mode requires --checkpoint <path>"
    echo ""
    echo "Latest checkpoints:"
    ls -t "$PROJECT_ROOT"/logs/WholeBodyTracking/*/model_*.pt 2>/dev/null | head -5 || echo "  (none found)"
    echo ""
    echo "Run '$0 --help' for usage."
    exit 1
fi

if [ "$MODE" = "hardware" ] && [ -z "$ONNX_PATH" ]; then
    echo "Error: --hardware mode requires --onnx <path>"
    echo ""
    echo "Export ONNX from checkpoint first:"
    echo "  $0 --checkpoint <model.pt> --sim  (runs eval and exports ONNX)"
    echo ""
    echo "Then use the exported ONNX:"
    echo "  $0 --onnx <checkpoint_dir>/exported/<name>.onnx --hardware"
    exit 1
fi

# ════════════════════════════════════════════════════════════════════
#  MODE 1: IsaacSim simulation
# ════════════════════════════════════════════════════════════════════
if [ "$MODE" = "sim" ]; then
    echo ""
    echo "=== G1-23DOF Evaluation in IsaacSim ==="
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Headless:   $HEADLESS"
    echo ""

    echo "Sourcing IsaacSim env..."
    unset CONDA_ENV_NAME
    source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"
    pip install -e "$PROJECT_ROOT/src/holosoma[unitree,booster]" --quiet

    # Check if ONNX already exported (also exports during eval)
    CHECKPOINT_DIR="$(dirname "$CHECKPOINT")"
    CHECKPOINT_STEM="$(basename "$CHECKPOINT" .pt)"
    ONNX_EXPORT_PATH="$CHECKPOINT_DIR/exported/${CHECKPOINT_STEM}.onnx"
    mkdir -p "$(dirname "$ONNX_EXPORT_PATH")"

    cd "$PROJECT_ROOT"
    echo "Launching IsaacSim evaluation..."
    echo "(Press Ctrl+C to stop)"
    echo ""

    python src/holosoma/holosoma/eval_agent.py \
        --checkpoint "$CHECKPOINT" \
        --training.export-onnx True \
        --training.num-envs "$NUM_ENVS" \
        --training.headless "$HEADLESS"

    echo ""
    if [ -f "$ONNX_EXPORT_PATH" ]; then
        echo "ONNX exported to: $ONNX_EXPORT_PATH"
        echo ""
        echo "To deploy on real robot:"
        echo "  $0 --onnx $ONNX_EXPORT_PATH --hardware"
    fi
    echo "Done."
    exit 0
fi

# ════════════════════════════════════════════════════════════════════
#  MODE 2: Real robot hardware
# ════════════════════════════════════════════════════════════════════
if [ "$MODE" = "hardware" ]; then
    if [ ! -f "$ONNX_PATH" ]; then
        echo "Error: ONNX file not found: $ONNX_PATH"
        exit 1
    fi

    echo ""
    echo "=== G1-23DOF Real Robot Deployment ==="
    echo "  ONNX:  $ONNX_PATH"
    echo ""
    echo "⚠️  WARNING: This will control the physical G1 robot!"
    echo "   Make sure the robot is in a safe position and E-stop is accessible."
    printf "   Continue? [y/N]: "
    read -r _confirm
    case "${_confirm}" in
        [Yy]*) echo "Proceeding..." ;;
        *) echo "Aborted."; exit 0 ;;
    esac

    echo ""
    echo "Sourcing inference env (hsinference)..."
    unset CONDA_ENV_NAME
    source "$PROJECT_ROOT/scripts/source_inference_setup.sh"
    pip install -e "$PROJECT_ROOT/src/holosoma_inference" --quiet

    echo ""
    echo "Keyboard controls (keep this terminal focused):"
    echo "  ]   Start policy"
    echo "  o   Stop policy"
    echo "  i   Default pose"
    echo "  s   Start motion clip"
    echo ""

    cd "$PROJECT_ROOT"
    python src/holosoma_inference/holosoma_inference/run_policy.py \
        inference:g1-23dof-wbt \
        --task.model-path "$ONNX_PATH"

    echo "Done."
    exit 0
fi
