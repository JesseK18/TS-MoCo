import os
import sys
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader
import pytorch_lightning as pl

# Import the global data loading from parent repo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from data.data_loader import data_loader
from utils_all.data_utils import loader_to_numpy

class UCRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        batch_size: int = 64,
        num_workers: int = 3,
        q_split: float = 0.15,
        seed: int = 18,
        permute_indexes: bool = False,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.q_split = q_split
        self.seed = seed
        self.permute_indexes = permute_indexes
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Use global data loading from parent repo
        dm_config = dict(
            data_dir=self.data_dir,
            dataset=self.dataset_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            q_split=self.q_split,
            seed=self.seed,
            permute_indexes=self.permute_indexes,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor
        )
        
        # Load data using global data loader
        dm = data_loader(dm_config)
        dm.prepare_data()
        dm.setup()
        
        # Extract numpy arrays from loaders
        train_loader = dm.train_dataloader()
        q_loader = dm.q_dataloader()
        test_loader = dm.test_dataloader()
        
        train_np, train_lbl_np = loader_to_numpy(train_loader)
        q_np, q_lbl_np = loader_to_numpy(q_loader)
        test_np, test_lbl_np = loader_to_numpy(test_loader)
        
        print(f"Data shapes after loading:")
        print(f"  Train: {train_np.shape}, labels: {train_lbl_np.shape}")
        print(f"  Q: {q_np.shape}, labels: {q_lbl_np.shape}")
        print(f"  Test: {test_np.shape}, labels: {test_lbl_np.shape}")
        
        # Convert to tensors
        x_train = torch.from_numpy(train_np).float()
        x_q = torch.from_numpy(q_np).float()
        x_test = torch.from_numpy(test_np).float()
        y_train = torch.from_numpy(train_lbl_np).long()
        y_q = torch.from_numpy(q_lbl_np).long()
        y_test = torch.from_numpy(test_lbl_np).long()

        # Handle shape conversion for TS-MoCo
        # Global loader returns (N, T, C) format, but our debug shows it's actually (N, C, T)
        # TS-MoCo expects (N, C, T) format
        print(f"Original data shapes:")
        print(f"  Train: {x_train.shape}")
        print(f"  Q: {x_q.shape}")
        print(f"  Test: {x_test.shape}")
        
        # Check if we need to permute based on actual shape
        if len(x_train.shape) == 3:
            # If shape is (N, T, C), convert to (N, C, T)
            if x_train.shape[1] > x_train.shape[2]:  # T > C, so it's (N, T, C)
                print("Converting from (N, T, C) to (N, C, T)")
                x_train = x_train.permute(0, 2, 1)  # -> (N, C, T)
                x_q = x_q.permute(0, 2, 1)
                x_test = x_test.permute(0, 2, 1)
            else:  # Already in (N, C, T) format
                print("Data already in (N, C, T) format, no permutation needed")
        elif len(x_train.shape) == 2:  # (N, T) - univariate
            print("Converting from (N, T) to (N, 1, T)")
            x_train = x_train.unsqueeze(1)  # -> (N, 1, T)
            x_q = x_q.unsqueeze(1)
            x_test = x_test.unsqueeze(1)
        
        print(f"Data shapes after permutation:")
        print(f"  Train: {x_train.shape}")
        print(f"  Q: {x_q.shape}")
        print(f"  Test: {x_test.shape}")
        
        # Create datasets
        self.train_dataset = TensorDataset(x_train, y_train)
        self.q_dataset = TensorDataset(x_q, y_q)
        self.test_dataset = TensorDataset(x_test, y_test)

        # Set input features and number of classes for TS-MoCo compatibility
        # Get sample from train dataset to determine input features
        sample_x, _ = self.train_dataset[0]
        print(f"Sample shape: {sample_x.shape}")
        
        if len(sample_x.shape) == 2:  # (C, T) format
            self.input_features = sample_x.shape[0]  # Number of features/channels
        elif len(sample_x.shape) == 1:  # (T,) format - univariate
            self.input_features = 1
        else:  # (T,) format
            self.input_features = 1
        
        # Get number of classes from unique labels
        all_labels = []
        for i in range(len(self.train_dataset)):
            _, label = self.train_dataset[i]
            all_labels.append(label.item())
        self.n_classes = len(set(all_labels))
        self.class_names = [f"Class_{i}" for i in range(self.n_classes)]
        
        print(f"TS-MoCo config:")
        print(f"  Input features: {self.input_features}")
        print(f"  Number of classes: {self.n_classes}")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Val samples: {len(self.q_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")

    def train_dataloader(self):
        # prefetch_factor only valid when num_workers>0
        if self.num_workers > 0:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                drop_last=False  # Don't drop incomplete batches
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=False  # Don't drop incomplete batches
            )

    def val_dataloader(self):
        # Use q_dataset as validation set (following your naming convention)
        if self.num_workers > 0:
            return DataLoader(
                self.q_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                drop_last=False  # Don't drop incomplete batches
            )
        else:
            return DataLoader(
                self.q_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False  # Don't drop incomplete batches
            )

    def test_dataloader(self):
        if self.num_workers > 0:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=self.persistent_workers,
                prefetch_factor=self.prefetch_factor,
                drop_last=False  # Don't drop incomplete batches
            )
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=False  # Don't drop incomplete batches
            )

    # Keep the original q_dataloader for backward compatibility
    def q_dataloader(self):
        return self.val_dataloader()


if __name__ == "__main__":
    # Test the datamodule
    datamodule = UCRDataModule(
        data_dir="../data/UCR",
        dataset_name="Coffee",  # Change to your dataset name
        batch_size=64,
        num_workers=1,
        q_split=0.15,
        seed=18,
        permute_indexes=False
    )
    datamodule.prepare_data()
    datamodule.setup(None)
    
    trainloader = datamodule.train_dataloader()
    valloader = datamodule.val_dataloader()
    testloader = datamodule.test_dataloader()
    
    for x, y in trainloader:
        print(f"Train - Input shape: {x.shape}")
        print(f"Train - Label shape: {y.shape}")
        print(f"Number of classes: {datamodule.n_classes}")
        print(f"Input features: {datamodule.input_features}")
        print(f"Unique labels: {torch.unique(y)}")
        break
    
    for x, y in valloader:
        print(f"Val - Input shape: {x.shape}")
        print(f"Val - Label shape: {y.shape}")
        break
    
    for x, y in testloader:
        print(f"Test - Input shape: {x.shape}")
        print(f"Test - Label shape: {y.shape}")
        break
