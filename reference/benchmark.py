#!/usr/bin/env python3
"""
xView3 ML Benchmark Script

This script runs comprehensive benchmarks to test GPU and HPC system performance.
It supports multiple configurations and generates detailed performance reports.

Usage:
    python benchmark.py --config benchmark_config.json
"""

import argparse
import json
import logging
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import platform
import psutil
import GPUtil
import torch

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class SystemInfo:
    """System information"""
    platform: str
    cpu_model: str
    cpu_cores: int
    cpu_logical_cores: int
    total_memory_gb: float
    gpu_count: int
    gpu_models: List[str]
    gpu_memory_gb: List[float]
    cuda_version: Optional[str] = None
    pytorch_version: Optional[str] = None


@dataclass
class OptimizationSettings:
    """Optimization settings for benchmarking"""
    use_amp: bool = False
    use_compile: bool = False
    compile_mode: str = "default"
    cudnn_benchmark: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    channels_last: bool = False


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    name: str
    batch_sizes: List[int]
    num_workers_list: List[int]
    epochs: int
    channels: str
    learning_rate: float
    distributed: bool
    world_size: int
    save_checkpoints: bool
    profile_memory: bool
    profile_gpu: bool
    optimizations: Optional[Dict[str, Any]] = None


@dataclass
class SingleRunConfig:
    """Configuration for a single benchmark run"""
    name: str
    batch_size: int
    num_workers: int
    epochs: int
    channels: str
    learning_rate: float
    distributed: bool
    world_size: int
    save_checkpoints: bool
    profile_memory: bool
    profile_gpu: bool
    optimizations: OptimizationSettings = None

    def __post_init__(self):
        if self.optimizations is None:
            self.optimizations = OptimizationSettings()


@dataclass
class BenchmarkResult:
    """Benchmark result"""
    config_name: str
    batch_size: int
    num_workers: int
    total_time: float
    epoch_times: List[float]
    avg_epoch_time: float
    throughput_samples_per_sec: float
    gpu_memory_peak_gb: Optional[float] = None
    gpu_utilization_avg: Optional[float] = None
    cpu_utilization_avg: Optional[float] = None
    final_loss: Optional[float] = None
    final_map: Optional[float] = None


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


def get_system_info() -> SystemInfo:
    """Get comprehensive system information"""
    logger = logging.getLogger(__name__)
    
    # CPU information
    cpu_info = platform.processor()
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_logical_cores = psutil.cpu_count(logical=True)
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # GPU information
    gpu_count = 0
    gpu_models = []
    gpu_memory_gb = []
    
    try:
        gpus = GPUtil.getGPUs()
        gpu_count = len(gpus)
        for gpu in gpus:
            gpu_models.append(gpu.name)
            gpu_memory_gb.append(gpu.memoryTotal / 1024)  # Convert to GB
    except Exception as e:
        logger.warning(f"Could not get GPU information: {e}")
    
    # CUDA version
    cuda_version = None
    try:
        import torch
        pytorch_version = torch.__version__
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
    except ImportError:
        logger.warning("PyTorch not available")
    
    return SystemInfo(
        platform=platform.platform(),
        cpu_model=cpu_info,
        cpu_cores=cpu_cores,
        cpu_logical_cores=cpu_logical_cores,
        total_memory_gb=total_memory_gb,
        gpu_count=gpu_count,
        gpu_models=gpu_models,
        gpu_memory_gb=gpu_memory_gb,
        cuda_version=cuda_version,
        pytorch_version=pytorch_version
    )


def load_benchmark_config(config_path: str) -> BenchmarkConfig:
    """Load benchmark configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    return BenchmarkConfig(**config_data)


def expand_config_to_runs(config: BenchmarkConfig) -> List[SingleRunConfig]:
    """Expand a BenchmarkConfig into individual SingleRunConfig for each batch_size/num_workers combo"""
    runs = []

    # Parse optimization settings if present
    opt_settings = OptimizationSettings()
    if config.optimizations:
        opt_settings = OptimizationSettings(
            use_amp=config.optimizations.get('use_amp', False),
            use_compile=config.optimizations.get('use_compile', False),
            compile_mode=config.optimizations.get('compile_mode', 'default'),
            cudnn_benchmark=config.optimizations.get('cudnn_benchmark', False),
            persistent_workers=config.optimizations.get('persistent_workers', False),
            prefetch_factor=config.optimizations.get('prefetch_factor', 2),
            channels_last=config.optimizations.get('channels_last', False),
        )

    for batch_size in config.batch_sizes:
        for num_workers in config.num_workers_list:
            runs.append(SingleRunConfig(
                name=f"{config.name}_bs{batch_size}_nw{num_workers}",
                batch_size=batch_size,
                num_workers=num_workers,
                epochs=config.epochs,
                channels=config.channels,
                learning_rate=config.learning_rate,
                distributed=config.distributed,
                world_size=config.world_size,
                save_checkpoints=config.save_checkpoints,
                profile_memory=config.profile_memory,
                profile_gpu=config.profile_gpu,
                optimizations=opt_settings,
            ))
    return runs


def run_training_benchmark(
    config: SingleRunConfig,
    data_paths: Dict[str, str],
    output_dir: str,
    logger: logging.Logger
) -> BenchmarkResult:
    """Run a single training benchmark"""
    
    # Create output directory for this benchmark
    benchmark_output_dir = os.path.join(output_dir, f"benchmark_{config.name}")
    os.makedirs(benchmark_output_dir, exist_ok=True)
    
    # Build training command
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    cmd = [
        sys.executable, train_script,
        "--train_data_root", data_paths["train_data_root"],
        "--val_data_root", data_paths["val_data_root"],
        "--train_label_file", data_paths["train_label_file"],
        "--val_label_file", data_paths["val_label_file"],
        "--train_chips_path", data_paths["train_chips_path"],
        "--val_chips_path", data_paths["val_chips_path"],
        "--channels", config.channels,
        "--epochs", str(config.epochs),
        "--batch_size", str(config.batch_size),
        "--num_workers", str(config.num_workers),
        "--learning_rate", str(config.learning_rate),
        "--output_dir", benchmark_output_dir,
        "--log_level", "INFO",
        "--print_freq", "100",  # Less verbose for benchmarking
        "--save_freq", str(config.epochs) if config.save_checkpoints else "999",
    ]
    
    if config.distributed:
        cmd.extend([
            "--distributed",
            "--world_size", str(config.world_size),
            "--dist_backend", "nccl",
            "--dist_url", "tcp://localhost:23456"
        ])

    # Add optimization flags
    if config.optimizations:
        opt = config.optimizations
        if opt.use_amp:
            cmd.append("--use_amp")
        if opt.use_compile:
            cmd.extend(["--use_compile", "--compile_mode", opt.compile_mode])
        if opt.cudnn_benchmark:
            cmd.append("--cudnn_benchmark")
        if opt.persistent_workers:
            cmd.extend(["--persistent_workers", "--prefetch_factor", str(opt.prefetch_factor)])
        if opt.channels_last:
            cmd.append("--channels_last")

    # Run training
    logger.info(f"Running benchmark: {config.name}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        total_time = time.time() - start_time
        
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            raise RuntimeError(f"Training failed with return code {result.returncode}")
        
        # Parse results from output
        epoch_times = []
        final_loss = None
        final_map = None
        
        for line in result.stdout.split('\n'):
            if "Epoch" in line and "Time:" in line:
                # Extract epoch time
                time_part = line.split("Time:")[1].split(",")[0].strip()
                epoch_time = float(time_part.replace("s", ""))
                epoch_times.append(epoch_time)
            
            if "Train Loss:" in line and "Val mAP:" in line:
                # Extract final metrics
                loss_part = line.split("Train Loss:")[1].split(",")[0].strip()
                map_part = line.split("Val mAP:")[1].split(",")[0].strip()
                final_loss = float(loss_part)
                final_map = float(map_part)
        
        # Calculate throughput
        # Estimate samples per epoch (this would need to be calculated from dataset size)
        samples_per_epoch = 1000  # This should be calculated from actual dataset
        throughput_samples_per_sec = (samples_per_epoch * config.epochs) / total_time
        
        # Get GPU memory usage if available
        gpu_memory_peak_gb = None
        gpu_utilization_avg = None
        
        if config.profile_gpu and torch.cuda.is_available():
            gpu_memory_peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()
        
        return BenchmarkResult(
            config_name=config.name,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            total_time=total_time,
            epoch_times=epoch_times,
            avg_epoch_time=sum(epoch_times) / len(epoch_times) if epoch_times else 0,
            throughput_samples_per_sec=throughput_samples_per_sec,
            gpu_memory_peak_gb=gpu_memory_peak_gb,
            gpu_utilization_avg=gpu_utilization_avg,
            final_loss=final_loss,
            final_map=final_map
        )
        
    except subprocess.TimeoutExpired:
        logger.error("Training timed out")
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def run_benchmark_suite(
    configs: List[BenchmarkConfig],
    data_paths: Dict[str, str],
    output_dir: str,
    logger: logging.Logger
) -> List[BenchmarkResult]:
    """Run the complete benchmark suite"""
    results = []

    # Expand all configs into individual runs
    all_runs = []
    for config in configs:
        all_runs.extend(expand_config_to_runs(config))

    logger.info(f"Expanded {len(configs)} configs into {len(all_runs)} individual runs")

    for i, run_config in enumerate(all_runs):
        logger.info(f"Running benchmark {i+1}/{len(all_runs)}: {run_config.name}")

        try:
            result = run_training_benchmark(run_config, data_paths, output_dir, logger)
            results.append(result)
            logger.info(f"Benchmark {run_config.name} completed successfully")

        except Exception as e:
            logger.error(f"Benchmark {run_config.name} failed: {e}")
            # Continue with next benchmark

    return results


def generate_benchmark_report(
    results: List[BenchmarkResult],
    system_info: SystemInfo,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """Generate comprehensive benchmark report"""
    
    # Create report
    report = {
        "system_info": asdict(system_info),
        "benchmark_results": [asdict(result) for result in results],
        "summary": {
            "total_benchmarks": len(results),
            "successful_benchmarks": len([r for r in results if r.total_time > 0]),
            "fastest_config": None,
            "highest_throughput": None,
            "best_accuracy": None,
        }
    }
    
    # Find best performers
    if results:
        fastest = min(results, key=lambda r: r.total_time)
        highest_throughput = max(results, key=lambda r: r.throughput_samples_per_sec)
        best_accuracy = max(results, key=lambda r: r.final_map or 0)
        
        report["summary"]["fastest_config"] = {
            "name": fastest.config_name,
            "batch_size": fastest.batch_size,
            "num_workers": fastest.num_workers,
            "total_time": fastest.total_time
        }
        
        report["summary"]["highest_throughput"] = {
            "name": highest_throughput.config_name,
            "batch_size": highest_throughput.batch_size,
            "num_workers": highest_throughput.num_workers,
            "throughput": highest_throughput.throughput_samples_per_sec
        }
        
        if best_accuracy.final_map:
            report["summary"]["best_accuracy"] = {
                "name": best_accuracy.config_name,
                "batch_size": best_accuracy.batch_size,
                "num_workers": best_accuracy.num_workers,
                "map": best_accuracy.final_map
            }
    
    # Save report
    report_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate markdown report
    markdown_report = generate_markdown_report(report)
    markdown_path = os.path.join(output_dir, "benchmark_report.md")
    with open(markdown_path, 'w') as f:
        f.write(markdown_report)
    
    logger.info(f"Benchmark report saved to {report_path}")
    logger.info(f"Markdown report saved to {markdown_path}")


def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generate markdown format report"""
    
    md = "# xView3 ML Benchmark Report\n\n"
    
    # System Information
    md += "## System Information\n\n"
    sys_info = report["system_info"]
    md += f"- **Platform**: {sys_info['platform']}\n"
    md += f"- **CPU**: {sys_info['cpu_model']} ({sys_info['cpu_cores']} cores, {sys_info['cpu_logical_cores']} logical)\n"
    md += f"- **Memory**: {sys_info['total_memory_gb']:.1f} GB\n"
    md += f"- **GPUs**: {sys_info['gpu_count']}\n"
    
    for i, (model, memory) in enumerate(zip(sys_info['gpu_models'], sys_info['gpu_memory_gb'])):
        md += f"  - GPU {i}: {model} ({memory:.1f} GB)\n"
    
    if sys_info['cuda_version']:
        md += f"- **CUDA**: {sys_info['cuda_version']}\n"
    if sys_info['pytorch_version']:
        md += f"- **PyTorch**: {sys_info['pytorch_version']}\n"
    
    md += "\n"
    
    # Summary
    md += "## Summary\n\n"
    summary = report["summary"]
    md += f"- **Total Benchmarks**: {summary['total_benchmarks']}\n"
    md += f"- **Successful**: {summary['successful_benchmarks']}\n\n"
    
    if summary['fastest_config']:
        fastest = summary['fastest_config']
        md += f"**Fastest Configuration**: {fastest['name']} "
        md += f"(batch_size={fastest['batch_size']}, num_workers={fastest['num_workers']}) "
        md += f"in {fastest['total_time']:.2f}s\n\n"
    
    if summary['highest_throughput']:
        highest = summary['highest_throughput']
        md += f"**Highest Throughput**: {highest['name']} "
        md += f"({highest['throughput']:.1f} samples/sec)\n\n"
    
    if summary['best_accuracy']:
        best = summary['best_accuracy']
        md += f"**Best Accuracy**: {best['name']} (mAP: {best['map']:.4f})\n\n"
    
    # Detailed Results
    md += "## Detailed Results\n\n"
    md += "| Config | Batch Size | Workers | Time (s) | Throughput | mAP |\n"
    md += "|--------|------------|---------|----------|------------|-----|\n"
    
    for result in report["benchmark_results"]:
        md += f"| {result['config_name']} | {result['batch_size']} | {result['num_workers']} | "
        md += f"{result['total_time']:.2f} | {result['throughput_samples_per_sec']:.1f} | "
        final_map = result.get('final_map')
        md += f"{final_map:.4f} |\n" if final_map is not None else "N/A |\n"
    
    return md


def create_default_configs() -> List[BenchmarkConfig]:
    """Create default benchmark configurations for different scenarios"""
    
    configs = []
    
    # Single GPU configurations
    configs.append(BenchmarkConfig(
        name="single_gpu_small_batch",
        batch_sizes=[4, 8],
        num_workers_list=[2, 4, 8],
        epochs=3,
        channels="vh,vv,bathymetry",
        learning_rate=0.005,
        distributed=False,
        world_size=1,
        save_checkpoints=False,
        profile_memory=True,
        profile_gpu=True
    ))
    
    configs.append(BenchmarkConfig(
        name="single_gpu_large_batch",
        batch_sizes=[16, 32],
        num_workers_list=[4, 8, 12],
        epochs=3,
        channels="vh,vv,bathymetry",
        learning_rate=0.005,
        distributed=False,
        world_size=1,
        save_checkpoints=False,
        profile_memory=True,
        profile_gpu=True
    ))
    
    # Multi-GPU configurations (if available)
    if torch.cuda.device_count() > 1:
        configs.append(BenchmarkConfig(
            name="multi_gpu_distributed",
            batch_sizes=[8, 16],
            num_workers_list=[4, 8],
            epochs=3,
            channels="vh,vv,bathymetry",
            learning_rate=0.005,
            distributed=True,
            world_size=torch.cuda.device_count(),
            save_checkpoints=False,
            profile_memory=True,
            profile_gpu=True
        ))
    
    return configs


def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(
        description="Run xView3 ML benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration arguments
    parser.add_argument('--config', type=str, default=None,
                       help='Path to benchmark configuration JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for benchmark results')
    
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
    
    # Benchmark arguments
    parser.add_argument('--use_default_configs', action='store_true',
                       help='Use default benchmark configurations')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting xView3 ML benchmark suite")
    
    # Get system information
    logger.info("Gathering system information...")
    system_info = get_system_info()
    logger.info(f"System: {system_info.cpu_cores} cores, {system_info.gpu_count} GPUs")
    
    # Load or create configurations
    if args.config:
        logger.info(f"Loading configurations from {args.config}")
        configs = [load_benchmark_config(args.config)]
    elif args.use_default_configs:
        logger.info("Using default benchmark configurations")
        configs = create_default_configs()
    else:
        logger.error("Must specify either --config or --use_default_configs")
        sys.exit(1)
    
    # Prepare data paths
    data_paths = {
        "train_data_root": args.train_data_root,
        "val_data_root": args.val_data_root,
        "train_label_file": args.train_label_file,
        "val_label_file": args.val_label_file,
        "train_chips_path": args.train_chips_path,
        "val_chips_path": args.val_chips_path,
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save system information
    system_info_path = os.path.join(args.output_dir, "system_info.json")
    with open(system_info_path, 'w') as f:
        json.dump(asdict(system_info), f, indent=2)
    
    # Run benchmarks
    logger.info(f"Running {len(configs)} benchmark configurations...")
    results = run_benchmark_suite(configs, data_paths, args.output_dir, logger)
    
    # Generate report
    logger.info("Generating benchmark report...")
    generate_benchmark_report(results, system_info, args.output_dir, logger)
    
    logger.info("Benchmark suite completed successfully!")


if __name__ == "__main__":
    main() 