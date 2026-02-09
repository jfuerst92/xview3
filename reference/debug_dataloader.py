import time
import torch
from dataloader import XView3Dataset
from utils import collate_fn

def test_dataloader_performance():
    """Test different DataLoader configurations to identify bottlenecks"""
    
    # Your current paths (adjust as needed)
    val_data_root = 'D:/xview3/tiny/validation'
    val_label_file = 'D:/xview3/labels/validation.csv'
    val_chips_path = 'D:/xview3/tiny/chips'
    channels = ['vh', 'vv', 'bathymetry']
    
    print("Creating dataset...")
    val_data = XView3Dataset(
        val_data_root,
        None,
        "val",
        chips_path=val_chips_path,
        detect_file=val_label_file,
        scene_list=None,
        background_frac=0.0,
        overwrite_preproc=False,
        channels=channels,
    )
    
    print(f"Dataset size: {len(val_data)}")
    
    # Test 1: Single sample loading (no workers)
    print("\n=== Test 1: Single sample loading ===")
    start_time = time.time()
    sample = val_data[0]
    single_time = time.time() - start_time
    print(f"Single sample load time: {single_time:.3f} seconds")
    
    # Test 2: DataLoader with 0 workers
    print("\n=== Test 2: DataLoader with 0 workers ===")
    val_sampler = torch.utils.data.SequentialSampler(val_data)
    data_loader_0 = torch.utils.data.DataLoader(
        val_data, batch_size=8, sampler=val_sampler, num_workers=0,
        collate_fn=collate_fn, pin_memory=True
    )
    
    start_time = time.time()
    batch = next(iter(data_loader_0))
    dl_0_time = time.time() - start_time
    print(f"DataLoader (0 workers) first batch time: {dl_0_time:.3f} seconds")
    
    # Test 3: DataLoader with 2 workers
    print("\n=== Test 3: DataLoader with 2 workers ===")
    data_loader_2 = torch.utils.data.DataLoader(
        val_data, batch_size=8, sampler=val_sampler, num_workers=2,
        collate_fn=collate_fn, pin_memory=True
    )
    
    start_time = time.time()
    batch = next(iter(data_loader_2))
    dl_2_time = time.time() - start_time
    print(f"DataLoader (2 workers) first batch time: {dl_2_time:.3f} seconds")
    
    # Test 4: DataLoader with 4 workers
    print("\n=== Test 4: DataLoader with 4 workers ===")
    data_loader_4 = torch.utils.data.DataLoader(
        val_data, batch_size=8, sampler=val_sampler, num_workers=4,
        collate_fn=collate_fn, pin_memory=True
    )
    
    start_time = time.time()
    batch = next(iter(data_loader_4))
    dl_4_time = time.time() - start_time
    print(f"DataLoader (4 workers) first batch time: {dl_4_time:.3f} seconds")
    
    # Test 5: DataLoader with 14 workers (your current setting)
    print("\n=== Test 5: DataLoader with 14 workers ===")
    data_loader_14 = torch.utils.data.DataLoader(
        val_data, batch_size=8, sampler=val_sampler, num_workers=14,
        collate_fn=collate_fn, pin_memory=True
    )
    
    start_time = time.time()
    batch = next(iter(data_loader_14))
    dl_14_time = time.time() - start_time
    print(f"DataLoader (14 workers) first batch time: {dl_14_time:.3f} seconds")
    
    # Analysis
    print("\n=== Analysis ===")
    print(f"Single sample load: {single_time:.3f}s")
    print(f"0 workers: {dl_0_time:.3f}s")
    print(f"2 workers: {dl_2_time:.3f}s")
    print(f"4 workers: {dl_4_time:.3f}s")
    print(f"14 workers: {dl_14_time:.3f}s")
    
    if dl_14_time > 60:  # More than 1 minute
        print("\n🚨 PROBLEM: 14 workers is taking too long!")
        print("Recommendation: Use 2-4 workers instead")
    elif dl_14_time > 10:  # More than 10 seconds
        print("\n⚠️ WARNING: 14 workers is slower than expected")
        print("Recommendation: Try 4-8 workers")
    else:
        print("\n✅ 14 workers is performing well")

if __name__ == "__main__":
    test_dataloader_performance() 