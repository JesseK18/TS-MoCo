#!/usr/bin/env python3
"""
Debug script to test classification with TS-MoCo embeddings.
"""

import os
import sys
import torch
import numpy as np

def debug_classification():
    print("Debugging TS-MoCo classification...")
    
    # Add parent directories to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))  # Go up to ICLR25 root
    sys.path.insert(0, parent_dir)
    sys.path.insert(0, current_dir)
    
    try:
        # Test imports
        print("Testing imports...")
        from datasets.ucr_dataset import UCRDataModule
        from architectures.TSMC import TSMC
        from functions.embeddings import compute_embeddings
        from classification.classification import eval_classification_emb
        
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
        
        # Create a simple TS-MoCo model
        print("Creating TS-MoCo model...")
        model = TSMC(
            pos_embeddings_alpha=0,
            input_features=datamodule.input_features,
            embedding_dim=128,
            n_head_token_enc=2,
            n_head_context_enc=2,
            depth_context_enc=1,
            max_predict_len=6
        )
        
        # Compute embeddings
        print("Computing embeddings...")
        train_loader = datamodule.train_dataloader()
        test_loader = datamodule.test_dataloader()
        
        train_emb, train_lbl = compute_embeddings(model, train_loader, device="cpu")
        test_emb, test_lbl = compute_embeddings(model, test_loader, device="cpu")
        
        print(f"Train embeddings shape: {train_emb.shape}")
        print(f"Train labels shape: {train_lbl.shape}")
        print(f"Test embeddings shape: {test_emb.shape}")
        print(f"Test labels shape: {test_lbl.shape}")
        
        print(f"Train embeddings type: {type(train_emb)}")
        print(f"Train labels type: {type(train_lbl)}")
        
        print(f"Train embeddings device: {train_emb.device}")
        print(f"Train labels device: {train_lbl.device}")
        
        # Test classification
        print("Testing classification...")
        try:
            acc, _ = eval_classification_emb(
                train_emb, train_lbl, 
                test_emb, test_lbl, 
                eval_protocol='linear'
            )
            print(f"✓ Classification successful! Accuracy: {acc}")
        except Exception as e:
            print(f"✗ Classification failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Try manual conversion
            print("Trying manual conversion...")
            train_emb_np = train_emb.numpy()
            train_lbl_np = train_lbl.numpy()
            test_emb_np = test_emb.numpy()
            test_lbl_np = test_lbl.numpy()
            
            print(f"Converted shapes:")
            print(f"  Train emb: {train_emb_np.shape}")
            print(f"  Train lbl: {train_lbl_np.shape}")
            print(f"  Test emb: {test_emb_np.shape}")
            print(f"  Test lbl: {test_lbl_np.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_classification()
    sys.exit(0 if success else 1)
