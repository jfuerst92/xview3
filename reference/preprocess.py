#!/usr/bin/env python3
"""
xView3 Data Preprocessing Script

This script preprocesses the xView3 dataset for training and benchmarking.
It creates chips from large satellite images and prepares annotations.

Usage:
    python preprocess.py --data_root /path/to/xview3/data --output_dir /path/to/output
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_dataloader import OptimizedXView3Dataset
from constants import BACKGROUND, FISHING, NONFISHING, NONVESSEL


def setup_logging(log_level: str, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration"""
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


def validate_paths(data_root: str, train_label_file: str, val_label_file: str) -> None:
    """Validate that all required paths exist"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root directory not found: {data_root}")
    
    if not os.path.exists(train_label_file):
        raise FileNotFoundError(f"Training label file not found: {train_label_file}")
    
    if not os.path.exists(val_label_file):
        raise FileNotFoundError(f"Validation label file not found: {val_label_file}")
    
    logger.info("All input paths validated successfully")


def get_scene_list(data_root: str, split: str) -> List[str]:
    """Get list of scene IDs for a given split"""
    split_dir = os.path.join(data_root, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    
    scenes = [
        d for d in os.listdir(split_dir) 
        if os.path.isdir(os.path.join(split_dir, d))
    ]
    return sorted(scenes)


def preprocess_split(
    data_root: str,
    label_file: str,
    chips_path: str,
    split: str,
    channels: List[str],
    num_workers: int,
    overwrite_preproc: bool,
    background_frac: Optional[float] = None,
    logger: logging.Logger = None
) -> None:
    """Preprocess a single split (train/val)"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info(f"Starting preprocessing for {split} split")
    start_time = time.time()
    
    # Create chips directory if it doesn't exist
    os.makedirs(chips_path, exist_ok=True)
    
    try:
        # Initialize dataset to trigger preprocessing
        dataset = OptimizedXView3Dataset(
            root=os.path.join(data_root, split),
            transforms=None,
            split=split,
            detect_file=label_file,
            scene_list=None,
            chips_path=chips_path,
            channels=channels,
            chip_size=800,
            overwrite_preproc=overwrite_preproc,
            bbox_size=5,
            background_frac=background_frac,
            background_min=3,
            ais_only=True,
            num_workers=num_workers,
            min_max_norm=True,
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed {split} preprocessing in {elapsed_time:.2f} seconds")
        logger.info(f"Number of chips created: {len(dataset)}")
        
        # Save dataset statistics
        stats = {
            'split': split,
            'num_chips': len(dataset),
            'channels': channels,
            'chip_size': 800,
            'preprocessing_time_seconds': elapsed_time,
            'num_workers': num_workers,
        }
        
        stats_file = os.path.join(chips_path, f'{split}_stats.json')
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved statistics to {stats_file}")
        
    except Exception as e:
        logger.error(f"Error preprocessing {split} split: {e}")
        raise


def main():
    """Main preprocessing function"""
    parser = argparse.ArgumentParser(
        description="Preprocess xView3 dataset for training and benchmarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing train/ and validation/ subdirectories'
    )
    parser.add_argument(
        '--train_label_file',
        type=str,
        required=True,
        help='Path to training labels CSV file'
    )
    parser.add_argument(
        '--val_label_file',
        type=str,
        required=True,
        help='Path to validation labels CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for preprocessed chips'
    )
    
    # Optional arguments
    parser.add_argument(
        '--channels',
        type=str,
        default='vh,vv,bathymetry',
        help='Comma-separated list of channels to process'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of workers for preprocessing'
    )
    parser.add_argument(
        '--overwrite_preproc',
        action='store_true',
        help='Overwrite existing preprocessed data'
    )
    parser.add_argument(
        '--background_frac',
        type=float,
        default=0.5,
        help='Fraction of background chips to include (for training only)'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Log file path (optional)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        default='train,val',
        help='Comma-separated list of splits to process'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting xView3 preprocessing")
    logger.info(f"Arguments: {vars(args)}")
    
    # Parse channels and splits
    channels = [c.strip() for c in args.channels.split(',')]
    splits = [s.strip() for s in args.splits.split(',')]
    
    # Validate inputs
    try:
        validate_paths(args.data_root, args.train_label_file, args.val_label_file)
    except FileNotFoundError as e:
        logger.error(f"Path validation failed: {e}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each split
    total_start_time = time.time()
    
    for split in splits:
        if split == 'train':
            label_file = args.train_label_file
            background_frac = args.background_frac
        elif split == 'val':
            label_file = args.val_label_file
            background_frac = None  # No background chips for validation
        else:
            logger.warning(f"Unknown split: {split}, skipping")
            continue
        
        split_output_dir = os.path.join(args.output_dir, split)
        
        try:
            preprocess_split(
                data_root=args.data_root,
                label_file=label_file,
                chips_path=split_output_dir,
                split=split,
                channels=channels,
                num_workers=args.num_workers,
                overwrite_preproc=args.overwrite_preproc,
                background_frac=background_frac,
                logger=logger
            )
        except Exception as e:
            logger.error(f"Failed to process {split} split: {e}")
            sys.exit(1)
    
    total_time = time.time() - total_start_time
    logger.info(f"Preprocessing completed in {total_time:.2f} seconds")
    
    # Save overall statistics
    overall_stats = {
        'total_time_seconds': total_time,
        'channels': channels,
        'num_workers': args.num_workers,
        'splits_processed': splits,
        'output_directory': args.output_dir,
    }
    
    stats_file = os.path.join(args.output_dir, 'preprocessing_stats.json')
    import json
    with open(stats_file, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    logger.info(f"Overall statistics saved to {stats_file}")
    logger.info("Preprocessing completed successfully!")


if __name__ == "__main__":
    main() 