#!/usr/bin/env python3
"""
Test script to verify TS-MoCo integration with global data loading.
"""

import os
import sys
import torch

def test_tsmoco_integration():
    print("Testing TS-MoCo integration...")
    
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Change to TS-MoCo directory
    tsmoco_dir = os.path.join(current_dir, 'models', 'tsmoco')
    os.chdir(tsmoco_dir)
    
    try:
        # Test imports
        print("Testing imports...")
        from datasets.ucr_dataset import UCRDataModule
        
        # Test data loading
        print("Testing data loading...")
        datamodule = UCRDataModule(
            data_dir="../../data",
            dataset_name="Coffee",
            batch_size=16,
            num_workers=0,  # Use 0 for testing
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
        
        # Test dataloaders
        print("Testing dataloaders...")
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        
        # Get a batch
        for batch in train_loader:
            x, y = batch
            print(f"✓ Dataloader successful!")
            print(f"  - Input shape: {x.shape}")
            print(f"  - Label shape: {y.shape}")
            print(f"  - Label range: {y.min().item()} to {y.max().item()}")
            break
        
        print("✓ All tests passed! TS-MoCo integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tsmoco_integration()
    sys.exit(0 if success else 1)
