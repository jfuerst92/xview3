# xView3 ML Benchmark Suite

A comprehensive, scalable ML benchmark system for testing GPU and HPC system performance using the xView3 satellite imagery dataset. This benchmark suite is designed to evaluate different hardware configurations including DGX A100 systems, GB200 GPUs, and laptop configurations.

## 🚀 Features

- **Multi-GPU Support**: Distributed training across multiple GPUs
- **Hardware Optimization**: Pre-configured settings for different systems
- **Comprehensive Benchmarking**: Automated performance testing with detailed reports
- **Scalable Architecture**: From single GPU laptops to multi-node HPC clusters
- **Reproducible Results**: Consistent benchmarking across different systems
- **Performance Profiling**: GPU memory usage, throughput, and accuracy metrics

## 📋 System Requirements

### Software Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU training)
- Required Python packages (see `requirements.txt`)

### Hardware Recommendations
- **Laptop RTX 3080Ti**: 16GB GPU memory, 32GB system RAM
- **GB200**: Single GB200 GPU with high memory capacity
- **DGX A100**: 8x A100 80GB GPUs for maximum performance

## 🏗️ Architecture

The benchmark suite consists of three main components:

### 1. Data Preprocessing (`preprocess.py`)
- Converts large satellite images into training chips
- Handles multi-channel data (VH, VV, bathymetry)
- Optimized for parallel processing
- Creates preprocessed datasets for efficient training

### 2. Training Engine (`train.py`)
- Faster R-CNN model for vessel detection
- Support for single and multi-GPU training
- Distributed training across multiple nodes
- Comprehensive logging and checkpointing

### 3. Benchmark Suite (`benchmark.py`)
- Automated performance testing
- Multiple configuration testing
- Detailed performance reports
- System information gathering

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd xview3-reference

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Preprocess the dataset
./run_benchmark.sh preprocess \
    --data-root /path/to/xview3/data \
    --train-labels /path/to/train.csv \
    --val-labels /path/to/validation.csv \
    --output-dir ./preprocessed_data
```

### 3. Run Benchmarks

#### For Laptop RTX 3080Ti:
```bash
./run_benchmark.sh laptop \
    --data-root /path/to/xview3/data \
    --train-labels /path/to/train.csv \
    --val-labels /path/to/validation.csv \
    --train-chips ./preprocessed_data/train \
    --val-chips ./preprocessed_data/val \
    --output-dir ./benchmarks/laptop
```

#### For DGX A100:
```bash
./run_benchmark.sh dgx-a100 \
    --data-root /path/to/xview3/data \
    --train-labels /path/to/train.csv \
    --val-labels /path/to/validation.csv \
    --train-chips ./preprocessed_data/train \
    --val-chips ./preprocessed_data/val \
    --output-dir ./benchmarks/dgx
```

#### For GB200:
```bash
./run_benchmark.sh gb200 \
    --data-root /path/to/xview3/data \
    --train-labels /path/to/train.csv \
    --val-labels /path/to/validation.csv \
    --train-chips ./preprocessed_data/train \
    --val-chips ./preprocessed_data/val \
    --output-dir ./benchmarks/gb200
```

## 📊 Configuration Files

The benchmark suite includes pre-configured settings for different hardware:

### `configs/laptop_rtx3080ti.json`
- Optimized for RTX 3080Ti laptops
- Batch sizes: 4, 8, 16
- Workers: 2, 4, 6, 8
- Single GPU training

### `configs/gb200.json`
- Optimized for GB200 systems
- Batch sizes: 8, 16, 32, 64
- Workers: 4, 8, 12, 16
- Single GPU training

### `configs/dgx_a100.json`
- Optimized for DGX A100 systems
- Batch sizes: 16, 32, 64, 128
- Workers: 8, 16, 24, 32
- Multi-GPU distributed training

## 📈 Performance Metrics

The benchmark suite measures:

- **Training Time**: Total time for complete training run
- **Throughput**: Samples processed per second
- **GPU Memory Usage**: Peak memory consumption
- **GPU Utilization**: Average GPU utilization
- **Accuracy**: Final mAP (mean Average Precision)
- **Epoch Times**: Individual epoch performance

## 📋 Output Structure

```
benchmarks/
├── system_info.json          # System specifications
├── benchmark_report.json     # Detailed results (JSON)
├── benchmark_report.md       # Human-readable report
└── benchmark_*/              # Individual benchmark results
    ├── args.json            # Configuration used
    ├── training_results.json # Training metrics
    ├── best_model.pth       # Best performing model
    └── checkpoints/         # Model checkpoints
```

## 🔧 Advanced Usage

### Custom Configuration

Create a custom benchmark configuration:

```json
{
  "name": "custom_benchmark",
  "batch_sizes": [8, 16, 32],
  "num_workers_list": [4, 8, 12],
  "epochs": 5,
  "channels": "vh,vv,bathymetry",
  "learning_rate": 0.005,
  "distributed": false,
  "world_size": 1,
  "save_checkpoints": false,
  "profile_memory": true,
  "profile_gpu": true
}
```

Run with custom configuration:

```bash
./run_benchmark.sh benchmark \
    --config configs/custom.json \
    --data-root /path/to/data \
    --train-labels /path/to/train.csv \
    --val-labels /path/to/validation.csv \
    --train-chips ./preprocessed_data/train \
    --val-chips ./preprocessed_data/val \
    --output-dir ./benchmarks/custom
```

### Single Training Run

For quick testing or development:

```bash
./run_benchmark.sh train \
    --data-root /path/to/data \
    --train-labels /path/to/train.csv \
    --val-labels /path/to/validation.csv \
    --train-chips ./preprocessed_data/train \
    --val-chips ./preprocessed_data/val \
    --output-dir ./training_run
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use fewer workers
   - Enable gradient checkpointing

2. **Slow Data Loading**
   - Increase `num_workers` (up to CPU core count)
   - Use SSD storage for data
   - Enable `pin_memory=True`

3. **Distributed Training Issues**
   - Ensure all nodes can communicate
   - Check firewall settings
   - Verify NCCL installation

### Performance Optimization

1. **For RTX 3080Ti Laptops**:
   - Use `num_workers=4-6` for optimal performance
   - Batch size 8-16 works well
   - Monitor thermal throttling

2. **For DGX A100**:
   - Use all 8 GPUs with distributed training
   - Large batch sizes (64-128) for efficiency
   - High worker counts (16-32)

3. **For GB200**:
   - Leverage high memory capacity
   - Experiment with large batch sizes
   - Optimize for throughput

## 📚 API Reference

### Preprocessing Script

```bash
python preprocess.py [OPTIONS]

Required:
  --data_root DIR           Root directory containing train/ and validation/
  --train_label_file FILE   Training labels CSV file
  --val_label_file FILE     Validation labels CSV file
  --output_dir DIR          Output directory for preprocessed data

Optional:
  --channels STR            Comma-separated channels (default: vh,vv,bathymetry)
  --num_workers INT         Number of preprocessing workers (default: 4)
  --overwrite_preproc       Overwrite existing preprocessed data
```

### Training Script

```bash
python train.py [OPTIONS]

Required:
  --train_data_root DIR     Training data root directory
  --val_data_root DIR       Validation data root directory
  --train_label_file FILE   Training labels CSV file
  --val_label_file FILE     Validation labels CSV file
  --train_chips_path DIR    Training chips directory
  --val_chips_path DIR      Validation chips directory
  --output_dir DIR          Output directory for results

Optional:
  --epochs INT              Number of training epochs (default: 10)
  --batch_size INT          Batch size per GPU (default: 8)
  --num_workers INT         Number of data loader workers (default: 4)
  --learning_rate FLOAT     Learning rate (default: 0.005)
  --distributed             Enable distributed training
```

### Benchmark Script

```bash
python benchmark.py [OPTIONS]

Required:
  --output_dir DIR          Output directory for benchmark results
  --train_data_root DIR     Training data root directory
  --val_data_root DIR       Validation data root directory
  --train_label_file FILE   Training labels CSV file
  --val_label_file FILE     Validation labels CSV file
  --train_chips_path DIR    Training chips directory
  --val_chips_path DIR      Validation chips directory

Optional:
  --config FILE             Custom benchmark configuration file
  --use_default_configs     Use default benchmark configurations
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- xView3 Challenge organizers for the dataset
- PyTorch team for the deep learning framework
- NVIDIA for GPU computing support

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration examples

---

**Note**: This benchmark suite is designed for research and development purposes. Results may vary based on system configuration, data quality, and environmental factors. 


grace notes:
  torch.empty(size, device='cuda', pin_memory=True)

    torch.cuda.set_per_process_memory_fraction(fraction, device)

     Grace Blackwell-Specific Environment Variables
Some NVIDIA environment variables can help tune performance:
CUDA_DEVICE_MAX_CONNECTIONS
CUDA_VISIBLE_DEVICES
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 (forces UM allocations to start on device)
These are advanced and usually not needed unless you’re tuning for maximum performance.