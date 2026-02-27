#!/bin/bash
#SBATCH --job-name=xview3-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1       # one srun task per node (torchrun handles per-GPU procs)
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32        # workers per node (tune to your cluster)
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ── User config ──────────────────────────────────────────────────────────────
DATA_ROOT=/data01/xview
SIF=/data01/xview/containers/xview3-train.sif   # path to Singularity SIF
OUTPUT_DIR=/data01/xview/output/run_${SLURM_JOB_ID}
EPOCHS=10
BATCH_SIZE=16       # per GPU; 16×16 GPUs = 256 effective batch
LR=0.02             # scale with batch: 0.005 × (256/4) ≈ 0.32, but cap conservatively
NUM_WORKERS=4       # DataLoader workers per GPU process
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs "$OUTPUT_DIR"

# Distributed rendezvous setup
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
NPROC_PER_NODE=8
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NODELIST"
echo "Master:       $MASTER_ADDR:$MASTER_PORT"
echo "World size:   $WORLD_SIZE  ($NNODES nodes × $NPROC_PER_NODE GPUs)"
echo "Output dir:   $OUTPUT_DIR"

# Bind mounts:
#   /data01/xview  → /data            (dataset)
#   $OUTPUT_DIR    → /output          (checkpoints)
BIND="$DATA_ROOT:/data,$OUTPUT_DIR:/output"

srun singularity exec \
    --nv \
    --bind "$BIND" \
    "$SIF" \
    torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$NPROC_PER_NODE" \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        /xview3/reference/train.py \
            --dataset_mode onthefly \
            --train_data_root /data/train \
            --val_data_root /data/validation \
            --train_label_file /data/labels/train.csv \
            --val_label_file /data/labels/validation.csv \
            --train_chips_path unused \
            --val_chips_path unused \
            --output_dir /output \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --learning_rate "$LR" \
            --num_workers "$NUM_WORKERS" \
            --use_amp \
            --distributed \
            --world_size "$WORLD_SIZE" \
            --log_file /output/train.log

# ── Alternate: pre-chipped mode (faster I/O if chips exist on shared FS) ─────
# Replace --dataset_mode onthefly with:
#   --dataset_mode prechipped \
#   --train_chips_path /data/chips/train \
#   --val_chips_path /data/chips/validation \
# ─────────────────────────────────────────────────────────────────────────────
