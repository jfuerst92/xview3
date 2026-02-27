#!/usr/bin/env python3
"""
Benchmark comparison: pre-chipped .npy loading vs on-the-fly GeoTIFF reading.

Selects ~20 scenes from each pool (chipped and raw) and measures:
  1. Data-only throughput (samples/sec iterating DataLoader, no GPU)
  2. Full training throughput (forward/backward with AMP + cudnn.benchmark)

Usage:
    python reference/benchmark_dataloader.py \
        --chipped_root D:/xview3/tiny \
        --chipped_chips D:/xview3/tiny/chips \
        --raw_root D:/xview3/validation \
        --label_file D:/xview3/labels/validation.csv \
        --output_dir benchmarks/dataloader_comparison \
        --num_scenes 20
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_dataloader import OptimizedXView3Dataset
from onthefly_dataloader import OnTheFlyXView3Dataset
from utils import collate_fn, xView3BaselineModel


def discover_chipped_scenes(chips_path, channels, max_scenes):
    """Find scenes that have pre-chipped .npy files."""
    first_ch = channels[0]
    scenes = []
    if not os.path.isdir(chips_path):
        return scenes
    for d in sorted(os.listdir(chips_path)):
        ch_dir = os.path.join(chips_path, d, first_ch)
        if os.path.isdir(ch_dir) and any(f.endswith(".npy") for f in os.listdir(ch_dir)):
            scenes.append(d)
        if len(scenes) >= max_scenes:
            break
    return scenes


def discover_raw_scenes(data_root, channels, max_scenes, exclude=None):
    """Find scenes that have raw GeoTIFFs."""
    tif_name = OnTheFlyXView3Dataset.CHANNEL_FILES[channels[0]]
    exclude = set(exclude or [])
    scenes = []
    if not os.path.isdir(data_root):
        return scenes
    for d in sorted(os.listdir(data_root)):
        if d in exclude:
            continue
        tif_path = os.path.join(data_root, d, tif_name)
        if os.path.isfile(tif_path):
            scenes.append(d)
        if len(scenes) >= max_scenes:
            break
    return scenes


def benchmark_dataonly(dataset, batch_size, num_workers, num_batches=50):
    """Iterate DataLoader without GPU, measure samples/sec."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        shuffle=False,
    )

    # Warmup
    it = iter(loader)
    for _ in range(min(3, len(loader))):
        try:
            next(it)
        except StopIteration:
            break

    total_samples = 0
    start = time.perf_counter()
    it = iter(loader)
    for i in range(num_batches):
        try:
            imgs, targets = next(it)
            total_samples += len(imgs)
        except StopIteration:
            break
    elapsed = time.perf_counter() - start

    return {
        "total_samples": total_samples,
        "elapsed_sec": round(elapsed, 3),
        "samples_per_sec": round(total_samples / elapsed, 2) if elapsed > 0 else 0,
    }


def benchmark_training(dataset, batch_size, num_workers, num_channels, epochs=3, device=None):
    """Full training loop with model forward/backward, AMP + cudnn.benchmark."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=True,
    )

    model = xView3BaselineModel(
        num_classes=4,
        num_channels=num_channels,
        image_mean=[0.5] * num_channels,
        image_std=[0.1] * num_channels,
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    use_amp = device.type == "cuda"

    epoch_times = []
    total_samples = 0

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        epoch_samples = 0

        for imgs, targets in loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    loss_dict = model(imgs, targets)
                    losses = sum(loss for loss in loss_dict.values())
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict = model(imgs, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()

            epoch_samples += len(imgs)

        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        total_samples += epoch_samples
        print(f"  Epoch {epoch}: {epoch_samples} samples in {epoch_time:.2f}s "
              f"({epoch_samples / epoch_time:.1f} samples/sec)")

    total_time = sum(epoch_times)
    return {
        "epochs": epochs,
        "total_samples": total_samples,
        "total_time_sec": round(total_time, 3),
        "epoch_times_sec": [round(t, 3) for t in epoch_times],
        "avg_samples_per_sec": round(total_samples / total_time, 2) if total_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark pre-chipped vs on-the-fly data loading")
    parser.add_argument("--chipped_root", type=str, required=True,
                        help="Root dir containing chipped scene folders (with raw TIFs)")
    parser.add_argument("--chipped_chips", type=str, required=True,
                        help="Chips directory for pre-chipped scenes")
    parser.add_argument("--raw_root", type=str, required=True,
                        help="Root dir containing raw scene folders (GeoTIFFs only)")
    parser.add_argument("--label_file", type=str, required=True,
                        help="Labels CSV covering both chipped and raw scenes")
    parser.add_argument("--output_dir", type=str, default="benchmarks/dataloader_comparison",
                        help="Output directory for results")
    parser.add_argument("--num_scenes", type=int, default=20,
                        help="Number of scenes to use from each pool")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--channels", type=str, default="vh,vv,bathymetry")
    parser.add_argument("--num_batches", type=int, default=50,
                        help="Number of batches for data-only benchmark")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs for training benchmark")
    parser.add_argument("--scene_cache_size", type=int, default=3,
                        help="Scene cache size for on-the-fly dataset")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip the full training benchmark (data-only)")
    args = parser.parse_args()

    channels = [c.strip() for c in args.channels.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover scenes
    print("Discovering scenes...")
    chipped_scenes = discover_chipped_scenes(args.chipped_chips, channels, args.num_scenes)
    raw_scenes = discover_raw_scenes(args.raw_root, channels, args.num_scenes,
                                     exclude=set(chipped_scenes))

    print(f"  Chipped scenes: {len(chipped_scenes)}")
    print(f"  Raw scenes: {len(raw_scenes)}")

    if not chipped_scenes:
        print("ERROR: No chipped scenes found. Check --chipped_chips path.")
        return
    if not raw_scenes:
        print("ERROR: No raw scenes found. Check --raw_root path.")
        return

    results = {"args": vars(args), "chipped_scenes": chipped_scenes, "raw_scenes": raw_scenes}

    # Create datasets
    print("\nCreating pre-chipped dataset...")
    ds_chipped = OptimizedXView3Dataset(
        root=args.chipped_root,
        transforms=None,
        split="train",
        detect_file=args.label_file,
        scene_list=chipped_scenes,
        chips_path=args.chipped_chips,
        channels=channels,
        chip_size=800,
        overwrite_preproc=False,
        bbox_size=5,
        background_frac=0.0,
        ais_only=True,
        num_workers=1,
        min_max_norm=True,
    )

    print("\nCreating on-the-fly dataset...")
    ds_onthefly = OnTheFlyXView3Dataset(
        root=args.raw_root,
        transforms=None,
        split="train",
        detect_file=args.label_file,
        scene_list=raw_scenes,
        channels=channels,
        chip_size=800,
        bbox_size=5,
        background_frac=0.0,
        ais_only=True,
        min_max_norm=True,
        scene_cache_size=args.scene_cache_size,
    )

    print(f"\n  Chipped dataset: {len(ds_chipped)} chips")
    print(f"  On-the-fly dataset: {len(ds_onthefly)} chips")

    # Verify output shapes match
    print("\nVerifying output shapes...")
    img_c, _ = ds_chipped[0]
    img_o, _ = ds_onthefly[0]
    print(f"  Chipped shape: {img_c.shape}")
    print(f"  On-the-fly shape: {img_o.shape}")
    assert img_c.shape == img_o.shape, f"Shape mismatch: {img_c.shape} vs {img_o.shape}"

    # Data-only benchmark
    print("\n=== Data-Only Benchmark ===")
    print(f"  Iterating {args.num_batches} batches, batch_size={args.batch_size}, "
          f"num_workers={args.num_workers}")

    print("\n  [Pre-chipped]")
    data_chipped = benchmark_dataonly(ds_chipped, args.batch_size, args.num_workers, args.num_batches)
    print(f"    {data_chipped['total_samples']} samples in {data_chipped['elapsed_sec']}s "
          f"= {data_chipped['samples_per_sec']} samples/sec")

    print("\n  [On-the-fly]")
    data_onthefly = benchmark_dataonly(ds_onthefly, args.batch_size, args.num_workers, args.num_batches)
    print(f"    {data_onthefly['total_samples']} samples in {data_onthefly['elapsed_sec']}s "
          f"= {data_onthefly['samples_per_sec']} samples/sec")

    results["data_only"] = {"prechipped": data_chipped, "onthefly": data_onthefly}

    if data_onthefly["samples_per_sec"] > 0:
        ratio = data_chipped["samples_per_sec"] / data_onthefly["samples_per_sec"]
        print(f"\n  Pre-chipped is {ratio:.2f}x faster for data loading")

    # Full training benchmark
    if not args.skip_training:
        print("\n=== Full Training Benchmark ===")
        print(f"  {args.epochs} epochs, batch_size={args.batch_size}, AMP + cudnn.benchmark")

        print("\n  [Pre-chipped]")
        train_chipped = benchmark_training(
            ds_chipped, args.batch_size, args.num_workers, len(channels), args.epochs
        )
        print(f"    Avg: {train_chipped['avg_samples_per_sec']} samples/sec")

        print("\n  [On-the-fly]")
        train_onthefly = benchmark_training(
            ds_onthefly, args.batch_size, args.num_workers, len(channels), args.epochs
        )
        print(f"    Avg: {train_onthefly['avg_samples_per_sec']} samples/sec")

        results["training"] = {"prechipped": train_chipped, "onthefly": train_onthefly}

        if train_onthefly["avg_samples_per_sec"] > 0:
            ratio = train_chipped["avg_samples_per_sec"] / train_onthefly["avg_samples_per_sec"]
            print(f"\n  Pre-chipped is {ratio:.2f}x faster for training")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Pre-chipped':>12} {'On-the-fly':>12}")
    print("-" * 60)
    print(f"{'Data-only (samples/sec)':<30} {data_chipped['samples_per_sec']:>12.1f} {data_onthefly['samples_per_sec']:>12.1f}")
    if not args.skip_training:
        print(f"{'Training (samples/sec)':<30} {train_chipped['avg_samples_per_sec']:>12.1f} {train_onthefly['avg_samples_per_sec']:>12.1f}")
    print("=" * 60)

    # Save results
    results_path = os.path.join(args.output_dir, "dataloader_benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
