#!/usr/bin/env python3
"""
Optimization Benchmark Runner

Runs training with different optimization combinations and compares performance.
Generates a markdown report with results.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Define optimization configurations to test
OPTIMIZATION_CONFIGS = [
    {
        "name": "baseline",
        "description": "No optimizations (baseline)",
        "flags": []
    },
    {
        "name": "amp",
        "description": "Automatic Mixed Precision (FP16)",
        "flags": ["--use_amp"]
    },
    {
        "name": "cudnn_benchmark",
        "description": "cudnn.benchmark auto-tuning",
        "flags": ["--cudnn_benchmark"]
    },
    {
        "name": "persistent_workers",
        "description": "Persistent DataLoader workers",
        "flags": ["--persistent_workers", "--prefetch_factor", "4"]
    },
    {
        "name": "amp_cudnn",
        "description": "AMP + cudnn.benchmark",
        "flags": ["--use_amp", "--cudnn_benchmark"]
    },
    {
        "name": "amp_cudnn_workers",
        "description": "AMP + cudnn + persistent workers",
        "flags": ["--use_amp", "--cudnn_benchmark", "--persistent_workers", "--prefetch_factor", "4"]
    },
    {
        "name": "all_optimizations",
        "description": "All optimizations (AMP + cudnn + workers + channels_last)",
        "flags": ["--use_amp", "--cudnn_benchmark", "--persistent_workers", "--prefetch_factor", "4", "--channels_last"]
    },
]


def run_training(args, config, output_dir):
    """Run a single training configuration"""
    config_output_dir = os.path.join(output_dir, config["name"])
    os.makedirs(config_output_dir, exist_ok=True)

    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "train.py"),
        "--train_data_root", args.train_data_root,
        "--val_data_root", args.val_data_root,
        "--train_label_file", args.train_label_file,
        "--val_label_file", args.val_label_file,
        "--train_chips_path", args.train_chips_path,
        "--val_chips_path", args.val_chips_path,
        "--channels", args.channels,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--output_dir", config_output_dir,
        "--print_freq", "50",
    ]

    # Add optimization flags
    cmd.extend(config["flags"])

    print(f"\n{'='*60}")
    print(f"Running: {config['name']} - {config['description']}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    total_time = time.time() - start_time

    # Parse results
    results_file = os.path.join(config_output_dir, "training_results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            training_results = json.load(f)
    else:
        training_results = {}

    return {
        "name": config["name"],
        "description": config["description"],
        "flags": config["flags"],
        "total_time": total_time,
        "training_results": training_results,
        "return_code": result.returncode,
        "stdout": result.stdout[-2000:] if result.stdout else "",  # Last 2000 chars
        "stderr": result.stderr[-1000:] if result.stderr else "",
    }


def generate_report(results, output_dir, args):
    """Generate markdown comparison report"""
    report_path = os.path.join(output_dir, "optimization_benchmark_report.md")

    # Find baseline time for comparison
    baseline_time = None
    for r in results:
        if r["name"] == "baseline" and r["return_code"] == 0:
            baseline_time = r["total_time"]
            break

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# ML Training Optimization Benchmark Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Test Configuration\n\n")
        f.write(f"- **Epochs**: {args.epochs}\n")
        f.write(f"- **Batch Size**: {args.batch_size}\n")
        f.write(f"- **Num Workers**: {args.num_workers}\n")
        f.write(f"- **Dataset**: {args.train_data_root}\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Optimization | Time (s) | Speedup | Status |\n")
        f.write("|-------------|----------|---------|--------|\n")

        for r in results:
            status = "✅" if r["return_code"] == 0 else "❌"
            time_str = f"{r['total_time']:.1f}"

            if baseline_time and r["return_code"] == 0:
                speedup = baseline_time / r["total_time"]
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "-"

            f.write(f"| {r['description']} | {time_str} | {speedup_str} | {status} |\n")

        f.write("\n## Detailed Results\n\n")

        for r in results:
            f.write(f"### {r['description']}\n\n")
            f.write(f"- **Config Name**: `{r['name']}`\n")
            f.write(f"- **Flags**: `{' '.join(r['flags']) if r['flags'] else 'None'}`\n")
            f.write(f"- **Total Time**: {r['total_time']:.2f}s\n")
            f.write(f"- **Return Code**: {r['return_code']}\n")

            if r["training_results"]:
                tr = r["training_results"]
                if "best_val_map" in tr:
                    f.write(f"- **Best Val mAP**: {tr['best_val_map']:.4f}\n")
                if "training_metrics" in tr and tr["training_metrics"]:
                    last_epoch = tr["training_metrics"][-1]
                    f.write(f"- **Final Train Loss**: {last_epoch.get('train_loss', 'N/A')}\n")

            f.write("\n")

        # Add recommendations
        f.write("## Recommendations\n\n")

        # Find best performing config
        successful = [r for r in results if r["return_code"] == 0]
        if successful:
            fastest = min(successful, key=lambda x: x["total_time"])
            f.write(f"**Fastest Configuration**: {fastest['description']}\n")
            f.write(f"- Time: {fastest['total_time']:.1f}s\n")
            if baseline_time:
                speedup = baseline_time / fastest['total_time']
                f.write(f"- Speedup vs baseline: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)\n")
            f.write(f"- Flags: `{' '.join(fastest['flags'])}`\n\n")

        f.write("### Key Takeaways\n\n")
        f.write("1. **AMP (Mixed Precision)** typically provides the largest single speedup\n")
        f.write("2. **cudnn.benchmark** helps when input sizes are consistent\n")
        f.write("3. **persistent_workers** reduces DataLoader overhead between epochs\n")
        f.write("4. Combining optimizations often provides cumulative benefits\n")

    print(f"\nReport saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Run optimization benchmarks")

    parser.add_argument('--train_data_root', type=str, required=True)
    parser.add_argument('--val_data_root', type=str, required=True)
    parser.add_argument('--train_label_file', type=str, required=True)
    parser.add_argument('--val_label_file', type=str, required=True)
    parser.add_argument('--train_chips_path', type=str, required=True)
    parser.add_argument('--val_chips_path', type=str, required=True)
    parser.add_argument('--channels', type=str, default='vh,vv,bathymetry')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./benchmarks/optimization_comparison')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                       help='Specific configs to run (default: all)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Select configs to run
    if args.configs:
        configs_to_run = [c for c in OPTIMIZATION_CONFIGS if c["name"] in args.configs]
    else:
        configs_to_run = OPTIMIZATION_CONFIGS

    print(f"Running {len(configs_to_run)} optimization configurations...")

    results = []
    for config in configs_to_run:
        result = run_training(args, config, args.output_dir)
        results.append(result)

        # Save intermediate results
        with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)

    # Generate report
    report_path = generate_report(results, args.output_dir, args)

    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)

    baseline_time = None
    for r in results:
        if r["name"] == "baseline" and r["return_code"] == 0:
            baseline_time = r["total_time"]
            break

    for r in results:
        status = "[OK]" if r["return_code"] == 0 else "[FAIL]"
        if baseline_time and r["return_code"] == 0:
            speedup = baseline_time / r["total_time"]
            print(f"{status} {r['name']:25s} {r['total_time']:8.1f}s  ({speedup:.2f}x)")
        else:
            print(f"{status} {r['name']:25s} {r['total_time']:8.1f}s")

    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()
