# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

xView3-Reference is an ML benchmark suite for vessel detection in Synthetic Aperture Radar (SAR) satellite imagery. It uses Faster R-CNN with ResNet-50 FPN backbone to detect and classify maritime vessels (fishing, non-fishing, non-vessel) in multi-channel SAR data from Sentinel-1.

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate xview3

# Add Jupyter kernel (optional)
pip install ipykernel
python -m ipykernel install --user --name xview3
```

## Common Commands

### Data Preprocessing
```bash
./reference/run_benchmark.sh preprocess \
    --data-root /path/to/xview3/data \
    --train-labels /path/to/train.csv \
    --val-labels /path/to/validation.csv \
    --output-dir ./preprocessed_data
```

### Training
```bash
# Single GPU
python reference/train.py \
    --train_data_root /data/train \
    --val_data_root /data/validation \
    --train_chips_path ./chips/train \
    --val_chips_path ./chips/val \
    --output_dir ./output \
    --epochs 10 --batch_size 8

# Multi-GPU distributed
python -m torch.distributed.launch --nproc_per_node=8 reference/train.py \
    --distributed --world_size 8 ...
```

### Inference
```bash
python reference/inference.py \
    --image_folder /path/to/scenes \
    --scene_ids "scene1,scene2" \
    --output predictions.csv \
    --weights ./reference/trained_model_3_epochs.pth \
    --chips_path ./chips \
    --channels vh vv bathymetry
```

### Scoring/Evaluation
```bash
python reference/metric.py \
    --inference_file predictions.csv \
    --label_file ground_truth.csv \
    --output results.json \
    --distance_tolerance 200 \
    --shore_tolerance 2 \
    --drop_low_detect --costly_dist
```

### Benchmarking
```bash
./reference/run_benchmark.sh laptop --data-root /data --output-dir ./benchmarks
./reference/run_benchmark.sh dgx-a100 --data-root /data --output-dir ./benchmarks
./reference/run_benchmark.sh gb200 --data-root /data --output-dir ./benchmarks
```

### Docker
```bash
# Build inference container
docker build -t xview3-inference:latest .

# Run inference in container
docker run --shm-size 16G --gpus=1 \
    --mount type=bind,source=/path/to/data,target=/on-docker/data \
    xview3-inference:latest /on-docker/data scene_id /on-docker/data/output.csv
```

## Architecture

### Data Pipeline
- Large satellite scenes are tiled into 800x800 pixel chips during preprocessing
- Multi-channel input: VH, VV, bathymetry, wind speed/direction, mask
- `OptimizedXView3Dataset` in `optimized_dataloader.py` handles training data loading
- `preprocess.py` orchestrates scene-to-chip conversion with annotation mapping

### Model
- Faster R-CNN with ResNet-50 FPN backbone (via torchvision)
- 3 output classes: fishing (1), non-fishing (2), non-vessel (3)
- Model creation in `utils.py:get_model()`
- Training loop in `engine.py`

### Distributed Training
- Uses PyTorch DistributedDataParallel (DDP)
- Rank-aware logging (only main process logs)
- Learning rate warmup for first 1000 iterations of first epoch

### Evaluation
- COCO-style mAP metrics via `coco_eval.py`
- xView3-specific scoring in `metric.py` with distance/shore tolerance

## Key Constants (constants.py)
```python
BACKGROUND = 0
FISHING = 1
NONFISHING = 2
NONVESSEL = 3
PIX_TO_M = 10  # 10 meters per pixel
```

## Directory Structure

```
reference/
├── train.py              # Main training script
├── inference.py          # Prediction script
├── benchmark.py          # Benchmarking suite
├── preprocess.py         # Data preprocessing
├── dataloader.py         # XView3Dataset
├── optimized_dataloader.py  # OptimizedXView3Dataset
├── engine.py             # Train/eval loops
├── utils.py              # Model, metrics, distributed utils
├── metric.py             # xView3 scoring metrics
├── coco_eval.py          # COCO evaluation
├── constants.py          # Class definitions
├── configs/              # Hardware benchmark configs
└── data_processing/      # GRD/OCN processing scripts
```

## Hardware Configs

Benchmark configurations in `reference/configs/`:
- `laptop_rtx3080ti.json`: batch 4-16, workers 2-8, single GPU
- `gb200.json`: batch 8-64, workers 4-16, single GPU
- `dgx_a100.json`: batch 16-128, workers 8-32, 8 GPUs distributed

## Notes

- Max inference time per scene: 15 minutes on V100 GPU
- No-data value for VH/VV/bathymetry: -32768
- Docker entrypoint requires exactly 3 positional args: image_folder, scene_ids, output
