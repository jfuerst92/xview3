#!/usr/bin/env python3
"""
xView3 Training Script

This script trains the xView3 Faster R-CNN model with support for:
- Single and multi-GPU training
- Distributed training across multiple nodes
- Comprehensive benchmarking and logging
- Configurable hyperparameters

Usage:
    python train.py --train_data_root /path/to/train --val_data_root /path/to/val
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_dataloader import OptimizedXView3Dataset
from onthefly_dataloader import OnTheFlyXView3Dataset, SceneGroupedSampler
from utils import collate_fn, xView3BaselineModel
from engine import train_one_epoch, evaluate


def setup_logging(log_level: str, log_file: Optional[str] = None, rank: int = 0) -> logging.Logger:
    """Setup logging configuration"""
    if rank != 0:  # Only log from main process in distributed training
        logging.basicConfig(level=logging.ERROR)
        return logging.getLogger(__name__)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )
    
    return logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, args) -> None:
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = args.dist_url.split(':')[0]
    os.environ['MASTER_PORT'] = args.dist_url.split(':')[1]
    
    # Initialize the process group
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=world_size,
        rank=rank
    )


def cleanup_distributed() -> None:
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(rank: int) -> torch.device:
    """Get device for current process"""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    return device


def create_datasets(args) -> tuple:
    """Create training and validation datasets"""
    logger = logging.getLogger(__name__)
    
    channels = [c.strip() for c in args.channels.split(',')]
    
    # Load scene lists if provided
    train_scene_list = None
    if args.train_scene_list:
        with open(args.train_scene_list) as f:
            train_scene_list = [line.strip() for line in f if line.strip()]
        logger.info(f"Using {len(train_scene_list)} training scenes from {args.train_scene_list}")

    val_scene_list = None
    if args.val_scene_list:
        with open(args.val_scene_list) as f:
            val_scene_list = [line.strip() for line in f if line.strip()]
        logger.info(f"Using {len(val_scene_list)} validation scenes from {args.val_scene_list}")

    if args.dataset_mode == "onthefly":
        logger.info(f"Creating ON-THE-FLY training dataset with channels: {channels}")
        train_data = OnTheFlyXView3Dataset(
            root=args.train_data_root,
            transforms=None,
            split="train",
            detect_file=args.train_label_file,
            scene_list=train_scene_list,
            channels=channels,
            chip_size=800,
            bbox_size=5,
            background_frac=args.background_frac,
            background_min=3,
            ais_only=True,
            min_max_norm=True,
            scene_cache_size=args.scene_cache_size,
        )

        logger.info(f"Creating ON-THE-FLY validation dataset with channels: {channels}")
        val_data = OnTheFlyXView3Dataset(
            root=args.val_data_root,
            transforms=None,
            split="val",
            detect_file=args.val_label_file,
            scene_list=val_scene_list,
            channels=channels,
            chip_size=800,
            bbox_size=5,
            background_frac=0.0,
            ais_only=True,
            min_max_norm=True,
            scene_cache_size=args.scene_cache_size,
        )
    else:
        logger.info(f"Creating training dataset with channels: {channels}")
        train_data = OptimizedXView3Dataset(
            root=args.train_data_root,
            transforms=None,
            split="train",
            detect_file=args.train_label_file,
            scene_list=train_scene_list,
            chips_path=args.train_chips_path,
            channels=channels,
            chip_size=800,
            overwrite_preproc=False,
            bbox_size=5,
            background_frac=args.background_frac,
            background_min=3,
            ais_only=True,
            num_workers=1,
            min_max_norm=True,
        )

        logger.info(f"Creating validation dataset with channels: {channels}")
        val_data = OptimizedXView3Dataset(
            root=args.val_data_root,
            transforms=None,
            split="val",
            detect_file=args.val_label_file,
            scene_list=val_scene_list,
            chips_path=args.val_chips_path,
            channels=channels,
            chip_size=800,
            overwrite_preproc=False,
            bbox_size=5,
            background_frac=0.0,
            background_min=3,
            ais_only=True,
            num_workers=1,
            min_max_norm=True,
        )
    
    logger.info(f"Training dataset size: {len(train_data)}")
    logger.info(f"Validation dataset size: {len(val_data)}")
    
    return train_data, val_data


def create_model(args, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Create and configure the model"""
    logger = logging.getLogger(__name__)
    
    # Load or compute image statistics
    # Use output_dir for stats storage (chips_path may be "unused" in onthefly mode)
    stats_path = args.output_dir
    if not os.path.exists(f'{stats_path}/data_means.npy'):
        logger.info("Computing image statistics...")
        image_mean = [0.5] * len(args.channels.split(','))
        image_std = [0.1] * len(args.channels.split(','))
        np.save(f'{stats_path}/data_means.npy', image_mean)
        np.save(f'{stats_path}/data_std.npy', image_std)
    else:
        logger.info("Loading existing image statistics...")
        image_mean = np.load(f'{stats_path}/data_means.npy')
        image_std = np.load(f'{stats_path}/data_std.npy')
    
    logger.info(f"Creating model with {num_classes} classes and {len(args.channels.split(','))} channels")
    model = xView3BaselineModel(
        num_classes=num_classes,
        num_channels=len(args.channels.split(',')),
        image_mean=image_mean,
        image_std=image_std,
    )
    
    model.to(device)
    return model


def create_data_loaders(
    train_data, val_data, args, rank: int, world_size: int
) -> tuple:
    """Create training and validation data loaders"""
    logger = logging.getLogger(__name__)
    
    if args.distributed:
        train_sampler = DistributedSampler(
            train_data, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_data, num_replicas=world_size, rank=rank, shuffle=False
        )
    elif args.dataset_mode == "onthefly":
        # Scene-grouped sampler: iterate all chips from each scene consecutively
        # so the LRU scene cache stays hot and avoids repeated 1+ GB TIF reloads
        train_sampler = SceneGroupedSampler(train_data.chip_indices, shuffle=True)
        val_sampler = SceneGroupedSampler(val_data.chip_indices, shuffle=False)
        logger.info("Using SceneGroupedSampler for on-the-fly data loading")
    else:
        train_sampler = torch.utils.data.RandomSampler(train_data)
        val_sampler = torch.utils.data.SequentialSampler(val_data)
    
    # DataLoader optimization settings
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
    }

    # Add persistent_workers if enabled and num_workers > 0
    if args.persistent_workers and args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = args.prefetch_factor

    train_loader = DataLoader(
        train_data,
        sampler=train_sampler,
        drop_last=True if args.distributed else False,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_data,
        sampler=val_sampler,
        **loader_kwargs
    )
    
    logger.info(f"Created data loaders with batch size {args.batch_size}")
    return train_loader, val_loader, train_sampler


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    args,
    metrics: Dict[str, Any],
    rank: int = 0
) -> None:
    """Save model checkpoint"""
    if rank != 0:  # Only save from main process
        return
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'args': vars(args),
        'metrics': metrics,
    }
    
    checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save latest checkpoint
    latest_path = os.path.join(args.output_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str,
    device: torch.device
) -> int:
    """Load model checkpoint"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(checkpoint_path):
        logger.info("No checkpoint found, starting from scratch")
        return 0
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    logger.info(f"Resuming from epoch {start_epoch}")
    
    return start_epoch


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    epoch: int,
    args,
    train_sampler=None,
    scaler=None
) -> Dict[str, float]:
    """Train for one epoch"""
    if train_sampler and hasattr(train_sampler, 'set_epoch'):
        train_sampler.set_epoch(epoch)

    return train_one_epoch(
        model, optimizer, train_loader, device, epoch,
        print_freq=args.print_freq,
        scaler=scaler,
        use_amp=args.use_amp,
        channels_last=args.channels_last
    )


def validate_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    args
) -> Dict[str, float]:
    """Validate for one epoch"""
    coco_evaluator = evaluate(model, val_loader, device=device)
    # Extract metrics from CocoEvaluator
    # stats[0] = AP @ IoU=0.50:0.95
    # stats[1] = AP @ IoU=0.50
    results = {}
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        results[iou_type] = {
            'AP': coco_eval.stats[0],
            'AP50': coco_eval.stats[1],
            'AP75': coco_eval.stats[2],
        }
    return results


def main_worker(rank: int, world_size: int, args) -> None:
    """Main worker function for distributed training"""
    # Setup distributed training
    if args.distributed:
        setup_distributed(rank, world_size, args)

    # Setup logging
    logger = setup_logging(args.log_level, args.log_file, rank)

    # Apply optimization settings
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cudnn.benchmark for auto-tuning")
    
    # Setup device
    device = get_device(rank)
    logger.info(f"Process {rank}/{world_size} using device: {device}")
    
    # Create datasets
    train_data, val_data = create_datasets(args)
    
    # Create data loaders
    train_loader, val_loader, train_sampler = create_data_loaders(
        train_data, val_data, args, rank, world_size
    )
    
    # Create model
    num_classes = len(np.unique(list(train_data.label_map.values())))
    model = create_model(args, num_classes, device)

    # Apply channels_last memory format if enabled
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        logger.info("Using channels_last memory format")

    # Apply torch.compile if enabled (PyTorch 2.0+)
    if args.use_compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
            logger.info(f"Applied torch.compile with mode={args.compile_mode}")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")

    # Wrap model for distributed training
    if args.distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # Setup AMP GradScaler if enabled
    scaler = None
    if args.use_amp:
        scaler = torch.amp.GradScaler('cuda')
        logger.info("Using Automatic Mixed Precision (AMP)")

    # Create optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(args.output_dir, 'latest_checkpoint.pth')
        start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path, device)
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    total_start_time = time.time()
    
    best_metric = 0.0
    training_metrics = []
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, optimizer, train_loader, device, epoch, args, train_sampler, scaler
        )
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, args)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_metrics.get('loss', 0.0),
            'val_map': val_metrics.get('bbox', {}).get('AP', 0.0),
            'epoch_time': epoch_time,
            'learning_rate': scheduler.get_last_lr()[0],
        }
        
        training_metrics.append(metrics)
        
        if rank == 0:  # Only log from main process
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {metrics['train_loss']:.4f}, "
                f"Val mAP: {metrics['val_map']:.4f}, "
                f"Time: {epoch_time:.2f}s, "
                f"LR: {metrics['learning_rate']:.6f}"
            )
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(model, optimizer, scheduler, epoch, args, metrics, rank)
        
        # Save best model
        if metrics['val_map'] > best_metric:
            best_metric = metrics['val_map']
            if rank == 0:
                best_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'metrics': metrics,
                }, best_path)
                logger.info(f"New best model saved with mAP: {best_metric:.4f}")
    
    total_time = time.time() - total_start_time
    
    # Save final results
    if rank == 0:
        results = {
            'total_training_time': total_time,
            'best_val_map': best_metric,
            'training_metrics': training_metrics,
            'args': vars(args),
        }
        
        results_path = os.path.join(args.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Best validation mAP: {best_metric:.4f}")
        logger.info(f"Results saved to {results_path}")
    
    # Cleanup
    if args.distributed:
        cleanup_distributed()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train xView3 Faster R-CNN model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train_data_root', type=str, required=True,
                       help='Training data root directory')
    parser.add_argument('--val_data_root', type=str, required=True,
                       help='Validation data root directory')
    parser.add_argument('--train_label_file', type=str, required=True,
                       help='Training labels CSV file')
    parser.add_argument('--val_label_file', type=str, required=True,
                       help='Validation labels CSV file')
    parser.add_argument('--train_chips_path', type=str, required=True,
                       help='Training chips directory')
    parser.add_argument('--val_chips_path', type=str, required=True,
                       help='Validation chips directory')
    parser.add_argument('--channels', type=str, default='vh,vv,bathymetry',
                       help='Comma-separated list of channels')
    parser.add_argument('--train_scene_list', type=str, default=None,
                       help='Text file with training scene IDs (one per line). If not provided, uses all scenes.')
    parser.add_argument('--val_scene_list', type=str, default=None,
                       help='Text file with validation scene IDs (one per line). If not provided, uses all scenes.')
    parser.add_argument('--background_frac', type=float, default=0.5,
                       help='Fraction of background chips for training')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                       help='Weight decay')
    parser.add_argument('--lr_step_size', type=int, default=3,
                       help='Learning rate step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                       help='Learning rate gamma')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--print_freq', type=int, default=10,
                       help='Print frequency during training')
    parser.add_argument('--save_freq', type=int, default=1,
                       help='Checkpoint save frequency')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Log file path')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Distributed backend')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:23456',
                       help='Distributed URL')
    parser.add_argument('--world_size', type=int, default=1,
                       help='Number of processes for distributed training')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')

    # Optimization flags for benchmarking
    parser.add_argument('--use_amp', action='store_true',
                       help='Use Automatic Mixed Precision (FP16)')
    parser.add_argument('--use_compile', action='store_true',
                       help='Use torch.compile() for model optimization (PyTorch 2.0+)')
    parser.add_argument('--compile_mode', type=str, default='default',
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='torch.compile mode')
    parser.add_argument('--cudnn_benchmark', action='store_true',
                       help='Enable cudnn.benchmark for auto-tuning')
    parser.add_argument('--persistent_workers', action='store_true',
                       help='Keep DataLoader workers alive between epochs')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                       help='Number of batches to prefetch per worker')
    parser.add_argument('--channels_last', action='store_true',
                       help='Use channels_last memory format for better GPU performance')

    # Dataset mode
    parser.add_argument('--dataset_mode', type=str, default='prechipped',
                       choices=['prechipped', 'onthefly'],
                       help='Dataset mode: prechipped (.npy chips) or onthefly (read GeoTIFFs directly)')
    parser.add_argument('--scene_cache_size', type=int, default=3,
                       help='Number of scenes to cache per worker in onthefly mode')

    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting xView3 training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Determine number of GPUs
    if args.distributed:
        world_size = args.world_size
    else:
        world_size = 1
    
    # Start training
    if args.distributed and world_size > 1:
        mp.spawn(
            main_worker,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        main_worker(0, 1, args)


if __name__ == "__main__":
    main()
