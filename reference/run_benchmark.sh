#!/bin/bash

# xView3 ML Benchmark Runner Script
# This script provides easy-to-use commands for running benchmarks on different systems

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATA_ROOT=""
TRAIN_LABELS=""
VAL_LABELS=""
TRAIN_CHIPS=""
VAL_CHIPS=""
OUTPUT_DIR=""
CONFIG=""

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "xView3 ML Benchmark Runner"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  preprocess    Preprocess the dataset"
    echo "  train         Run single training run"
    echo "  benchmark     Run comprehensive benchmark suite"
    echo "  dgx-a100      Run DGX A100 optimized benchmark"
    echo "  gb200         Run GB200 optimized benchmark"
    echo "  laptop        Run laptop RTX 3080Ti benchmark"
    echo ""
    echo "Options:"
    echo "  --data-root DIR       Root directory containing train/ and validation/"
    echo "  --train-labels FILE   Training labels CSV file"
    echo "  --val-labels FILE     Validation labels CSV file"
    echo "  --train-chips DIR     Training chips directory"
    echo "  --val-chips DIR       Validation chips directory"
    echo "  --output-dir DIR      Output directory for results"
    echo "  --config FILE         Custom benchmark configuration file"
    echo ""
    echo "Examples:"
    echo "  $0 preprocess --data-root /path/to/data --output-dir ./preprocessed"
    echo "  $0 dgx-a100 --data-root /path/to/data --output-dir ./benchmarks/dgx"
    echo "  $0 benchmark --config configs/custom.json --output-dir ./benchmarks"
}

# Function to validate required arguments
validate_args() {
    if [[ -z "$DATA_ROOT" ]]; then
        print_error "Data root directory is required (--data-root)"
        exit 1
    fi
    
    if [[ -z "$TRAIN_LABELS" ]]; then
        print_error "Training labels file is required (--train-labels)"
        exit 1
    fi
    
    if [[ -z "$VAL_LABELS" ]]; then
        print_error "Validation labels file is required (--val-labels)"
        exit 1
    fi
    
    if [[ -z "$TRAIN_CHIPS" ]]; then
        print_error "Training chips directory is required (--train-chips)"
        exit 1
    fi
    
    if [[ -z "$VAL_CHIPS" ]]; then
        print_error "Validation chips directory is required (--val-chips)"
        exit 1
    fi
    
    if [[ -z "$OUTPUT_DIR" ]]; then
        print_error "Output directory is required (--output-dir)"
        exit 1
    fi
}

# Function to run preprocessing
run_preprocess() {
    print_info "Running data preprocessing..."
    
    python preprocess.py \
        --data_root "$DATA_ROOT" \
        --train_label_file "$TRAIN_LABELS" \
        --val_label_file "$VAL_LABELS" \
        --output_dir "$OUTPUT_DIR" \
        --channels "vh,vv,bathymetry" \
        --num_workers 4 \
        --log_level INFO
    
    print_success "Preprocessing completed!"
}

# Function to run single training
run_train() {
    print_info "Running single training run..."
    
    python train.py \
        --train_data_root "$DATA_ROOT/train" \
        --val_data_root "$DATA_ROOT/validation" \
        --train_label_file "$TRAIN_LABELS" \
        --val_label_file "$VAL_LABELS" \
        --train_chips_path "$TRAIN_CHIPS" \
        --val_chips_path "$VAL_CHIPS" \
        --output_dir "$OUTPUT_DIR" \
        --epochs 5 \
        --batch_size 8 \
        --num_workers 4 \
        --log_level INFO
    
    print_success "Training completed!"
}

# Function to run benchmark
run_benchmark() {
    print_info "Running benchmark suite..."
    
    if [[ -n "$CONFIG" ]]; then
        print_info "Using custom configuration: $CONFIG"
        python benchmark.py \
            --config "$CONFIG" \
            --train_data_root "$DATA_ROOT/train" \
            --val_data_root "$DATA_ROOT/validation" \
            --train_label_file "$TRAIN_LABELS" \
            --val_label_file "$VAL_LABELS" \
            --train_chips_path "$TRAIN_CHIPS" \
            --val_chips_path "$VAL_CHIPS" \
            --output_dir "$OUTPUT_DIR" \
            --log_level INFO
    else
        print_info "Using default configurations"
        python benchmark.py \
            --use_default_configs \
            --train_data_root "$DATA_ROOT/train" \
            --val_data_root "$DATA_ROOT/validation" \
            --train_label_file "$TRAIN_LABELS" \
            --val_label_file "$VAL_LABELS" \
            --train_chips_path "$TRAIN_CHIPS" \
            --val_chips_path "$VAL_CHIPS" \
            --output_dir "$OUTPUT_DIR" \
            --log_level INFO
    fi
    
    print_success "Benchmark completed!"
}

# Function to run DGX A100 benchmark
run_dgx_a100() {
    print_info "Running DGX A100 optimized benchmark..."
    
    # Check if we have multiple GPUs
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [[ $GPU_COUNT -lt 8 ]]; then
        print_warning "Detected $GPU_COUNT GPUs, but DGX A100 typically has 8 GPUs"
        print_warning "Consider using 'laptop' or 'gb200' configuration instead"
    fi
    
    python benchmark.py \
        --config "configs/dgx_a100.json" \
        --train_data_root "$DATA_ROOT/train" \
        --val_data_root "$DATA_ROOT/validation" \
        --train_label_file "$TRAIN_LABELS" \
        --val_label_file "$VAL_LABELS" \
        --train_chips_path "$TRAIN_CHIPS" \
        --val_chips_path "$VAL_CHIPS" \
        --output_dir "$OUTPUT_DIR" \
        --log_level INFO
    
    print_success "DGX A100 benchmark completed!"
}

# Function to run GB200 benchmark
run_gb200() {
    print_info "Running GB200 optimized benchmark..."
    
    python benchmark.py \
        --config "configs/gb200.json" \
        --train_data_root "$DATA_ROOT/train" \
        --val_data_root "$DATA_ROOT/validation" \
        --train_label_file "$TRAIN_LABELS" \
        --val_label_file "$VAL_LABELS" \
        --train_chips_path "$TRAIN_CHIPS" \
        --val_chips_path "$VAL_CHIPS" \
        --output_dir "$OUTPUT_DIR" \
        --log_level INFO
    
    print_success "GB200 benchmark completed!"
}

# Function to run laptop benchmark
run_laptop() {
    print_info "Running laptop RTX 3080Ti optimized benchmark..."
    
    python benchmark.py \
        --config "configs/laptop_rtx3080ti.json" \
        --train_data_root "$DATA_ROOT/train" \
        --val_data_root "$DATA_ROOT/validation" \
        --train_label_file "$TRAIN_LABELS" \
        --val_label_file "$VAL_LABELS" \
        --train_chips_path "$TRAIN_CHIPS" \
        --val_chips_path "$VAL_CHIPS" \
        --output_dir "$OUTPUT_DIR" \
        --log_level INFO
    
    print_success "Laptop benchmark completed!"
}

# Parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        preprocess|train|benchmark|dgx-a100|gb200|laptop)
            COMMAND="$1"
            shift
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --train-labels)
            TRAIN_LABELS="$2"
            shift 2
            ;;
        --val-labels)
            VAL_LABELS="$2"
            shift 2
            ;;
        --train-chips)
            TRAIN_CHIPS="$2"
            shift 2
            ;;
        --val-chips)
            VAL_CHIPS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if command was provided
if [[ -z "$COMMAND" ]]; then
    print_error "No command specified"
    show_usage
    exit 1
fi

# Validate arguments based on command
case $COMMAND in
    preprocess)
        if [[ -z "$DATA_ROOT" || -z "$TRAIN_LABELS" || -z "$VAL_LABELS" || -z "$OUTPUT_DIR" ]]; then
            print_error "Missing required arguments for preprocessing"
            show_usage
            exit 1
        fi
        run_preprocess
        ;;
    train)
        validate_args
        run_train
        ;;
    benchmark)
        validate_args
        run_benchmark
        ;;
    dgx-a100)
        validate_args
        run_dgx_a100
        ;;
    gb200)
        validate_args
        run_gb200
        ;;
    laptop)
        validate_args
        run_laptop
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac 