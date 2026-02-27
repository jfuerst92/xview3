#!/bin/bash
# Run xView3 training on a single multi-GPU node without SLURM.
#
# Usage:
#   ./scripts/run_single_node.sh [--gpus N] [--batch-size N] [--epochs N] [--output-dir DIR]
#
# Examples:
#   ./scripts/run_single_node.sh                        # 8 GPUs, defaults
#   ./scripts/run_single_node.sh --gpus 4               # 4 GPUs
#   ./scripts/run_single_node.sh --gpus 1 --batch-size 4  # single GPU debug run
#
# Container usage (Docker):
#   docker run --gpus all --shm-size=64G \
#       --mount type=bind,source=/data01/xview,target=/data \
#       xview3-train:latest \
#       bash /xview3/scripts/run_single_node.sh --data-root /data
#
# Container usage (Singularity):
#   singularity exec --nv --bind /data01/xview:/data xview3-train.sif \
#       bash /xview3/scripts/run_single_node.sh --data-root /data

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
GPUS=8
DATA_ROOT=/data01/xview
OUTPUT_DIR="${DATA_ROOT}/output/run_$(date +%Y%m%d_%H%M%S)"
EPOCHS=10
BATCH_SIZE=16          # per GPU
NUM_WORKERS=4          # DataLoader workers per GPU process
DATASET_MODE=onthefly  # or: prechipped
SCENE_CACHE_SIZE=3
# LR scales linearly with effective batch: base 0.005 @ batch 4 × N GPUs × batch_size
# Computed below automatically; override with --lr if needed
LR=""
# ─────────────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --gpus N              Number of GPUs to use (default: $GPUS)
  --data-root DIR       Root data directory (default: $DATA_ROOT)
  --output-dir DIR      Output directory for checkpoints/logs (default: auto)
  --epochs N            Number of training epochs (default: $EPOCHS)
  --batch-size N        Batch size per GPU (default: $BATCH_SIZE)
  --num-workers N       DataLoader workers per process (default: $NUM_WORKERS)
  --dataset-mode MODE   prechipped or onthefly (default: $DATASET_MODE)
  --lr LR               Learning rate (default: auto-scaled)
  -h, --help            Show this help
EOF
    exit 0
}

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)           GPUS=$2;           shift 2 ;;
        --data-root)      DATA_ROOT=$2;      shift 2 ;;
        --output-dir)     OUTPUT_DIR=$2;     shift 2 ;;
        --epochs)         EPOCHS=$2;         shift 2 ;;
        --batch-size)     BATCH_SIZE=$2;     shift 2 ;;
        --num-workers)    NUM_WORKERS=$2;    shift 2 ;;
        --dataset-mode)   DATASET_MODE=$2;   shift 2 ;;
        --lr)             LR=$2;             shift 2 ;;
        -h|--help)        usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Derived config ────────────────────────────────────────────────────────────
WORLD_SIZE=$GPUS

# Linear LR scaling: base 0.005 at effective batch 4, capped at 0.04
if [[ -z "$LR" ]]; then
    EFFECTIVE_BATCH=$((BATCH_SIZE * WORLD_SIZE))
    LR=$(python3 -c "lr = min(0.005 * $EFFECTIVE_BATCH / 4, 0.04); print(f'{lr:.4f}')")
fi

# Chips paths — only used in prechipped mode
if [[ "$DATASET_MODE" == "prechipped" ]]; then
    TRAIN_CHIPS="${DATA_ROOT}/chips/train"
    VAL_CHIPS="${DATA_ROOT}/chips/validation"
else
    TRAIN_CHIPS="unused"
    VAL_CHIPS="unused"
fi

mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════"
echo " xView3 Single-Node Training"
echo "════════════════════════════════════════"
echo " GPUs:            $GPUS"
echo " Dataset mode:    $DATASET_MODE"
echo " Data root:       $DATA_ROOT"
echo " Output dir:      $OUTPUT_DIR"
echo " Epochs:          $EPOCHS"
echo " Batch/GPU:       $BATCH_SIZE"
echo " Effective batch: $((BATCH_SIZE * WORLD_SIZE))"
echo " Learning rate:   $LR"
echo " Workers/proc:    $NUM_WORKERS"
echo "════════════════════════════════════════"

# ── Launch ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

torchrun \
    --standalone \
    --nproc_per_node="$GPUS" \
    "$REPO_ROOT/reference/train.py" \
        --dataset_mode "$DATASET_MODE" \
        --scene_cache_size "$SCENE_CACHE_SIZE" \
        --train_data_root "${DATA_ROOT}/train" \
        --val_data_root "${DATA_ROOT}/validation" \
        --train_label_file "${DATA_ROOT}/labels/train.csv" \
        --val_label_file "${DATA_ROOT}/labels/validation.csv" \
        --train_chips_path "$TRAIN_CHIPS" \
        --val_chips_path "$VAL_CHIPS" \
        --output_dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LR" \
        --num_workers "$NUM_WORKERS" \
        --use_amp \
        --distributed \
        --world_size "$WORLD_SIZE" \
        --log_file "${OUTPUT_DIR}/train.log"
