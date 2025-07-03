"""
Debug script to test GPU memory usage during dataset loading
"""
import torch
import psutil
import os


def print_memory_usage(stage=""):
    """Print current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / 1024 / 1024 / 1024
    
    # GPU memory if available
    gpu_mem = ""
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        gpu_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
        gpu_mem = f" | GPU: {gpu_allocated:.2f}GB allocated, {gpu_reserved:.2f}GB reserved"
    
    print(f"[{stage}] Memory usage: {memory_gb:.2f}GB RAM{gpu_mem}")


if __name__ == "__main__":
    print_memory_usage("Script start")
    
    # Import your modules
    from dpvtex.dpvt_data import train_val_data_from_preprocessed
    print_memory_usage("After imports")
    
    # Load your dataset
    data_name = "influenzaC_fluC_PB2_tree_search"
    device = "cuda"
    data_nicknames_path = "train/data_nicknames.json"
    
    train_data, val_data = train_val_data_from_preprocessed(data_name, device, data_nicknames_path)
    print_memory_usage("After loading datasets")
    
    # Test accessing a few samples
    print("Testing sample access...")
    for i in range(3):
        sample = train_data[i]
        print_memory_usage(f"After accessing sample {i}")
    
    print("Testing DataLoader creation...")
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    print_memory_usage("After DataLoader creation")
    
    print("Testing first batch...")
    batch = next(iter(train_loader))
    print_memory_usage("After loading first batch")
    
    print("Batch shapes:")
    for i, tensor in enumerate(batch):
        print(f"  Tensor {i}: {tensor.shape} on {tensor.device}")