#!/usr/bin/env python3
"""
Debug script to test TS-MoCo data loading and identify shape issues.
"""

import os
import sys
import torch
import numpy as np

def debug_tsmoco_data():
    print("Debugging TS-MoCo data loading...")
    
    # Add parent directories to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up to ICLR25 root
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, current_dir)
    
    try:
        # Test imports
        print("Testing imports...")
        from datasets.ucr_dataset import UCRDataModule
        
        # Test data loading with Coffee dataset
        print("Testing data loading with Coffee dataset...")
        datamodule = UCRDataModule(
            data_dir="../../data",
            dataset_name="Coffee",
            batch_size=4,  # Small batch size for debugging
            num_workers=0,  # Use 0 for debugging
            q_split=0.15,
            seed=18,
            permute_indexes=True
        )
        
        datamodule.prepare_data()
        datamodule.setup()
        
        print(f"✓ Data loading successful!")
        print(f"  - Input features: {datamodule.input_features}")
        print(f"  - Number of classes: {datamodule.n_classes}")
        print(f"  - Train dataset size: {len(datamodule.train_dataset)}")
        print(f"  - Val dataset size: {len(datamodule.q_dataset)}")
        print(f"  - Test dataset size: {len(datamodule.test_dataset)}")
        
        # Test dataloaders and check batch shapes
        print("Testing dataloaders...")
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        
        # Check first few batches
        print("Checking train batches:")
        for i, batch in enumerate(train_loader):
            x, y = batch
            print(f"  Batch {i}: x.shape={x.shape}, y.shape={y.shape}")
            print(f"    x min/max: {x.min().item():.3f}/{x.max().item():.3f}")
            print(f"    y unique: {torch.unique(y).tolist()}")
            if i >= 2:  # Check first 3 batches
                break
        
        print("Checking val batches:")
        for i, batch in enumerate(val_loader):
            x, y = batch
            print(f"  Batch {i}: x.shape={x.shape}, y.shape={y.shape}")
            if i >= 1:  # Check first 2 batches
                break
        
        print("Checking test batches:")
        for i, batch in enumerate(test_loader):
            x, y = batch
            print(f"  Batch {i}: x.shape={x.shape}, y.shape={y.shape}")
            if i >= 1:  # Check first 2 batches
                break
        
        # Test individual samples
        print("Testing individual samples:")
        for i in range(min(3, len(datamodule.train_dataset))):
            x, y = datamodule.train_dataset[i]
            print(f"  Sample {i}: x.shape={x.shape}, y={y.item()}")
        
        print("✓ All data loading tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_tsmoco_data()
    sys.exit(0 if success else 1)
