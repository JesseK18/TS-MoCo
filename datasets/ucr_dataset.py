import os
import sys
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split, DataLoader
import pytorch_lightning as pl

# Import the user's UCR preprocessing
from .UCR_preprocess import load_UCR

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

        # prepare cache path
        root_dir = os.path.dirname(os.path.dirname(__file__))
        cache_dir = os.path.join(root_dir, 'data', 'UCR_cache')
        os.makedirs(cache_dir, exist_ok=True)
        suffix = '_perm' if permute_indexes else ''
        self.cache_path = os.path.join(cache_dir, f'{self.dataset_name}{suffix}.pt')

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if os.path.exists(self.cache_path):
            # load cached tensors
            data = torch.load(self.cache_path)
            x_train, y_train = data['x_train'], data['y_train']
            x_q, y_q = data['x_q'], data['y_q']
            x_test, y_test = data['x_test'], data['y_test']
            
            # wrap into Datasets
            self.train_dataset = TensorDataset(x_train, y_train)
            self.q_dataset = TensorDataset(x_q, y_q)
            self.test_dataset = TensorDataset(x_test, y_test)
        
        else:
            # load with ts2vec logic
            train_np, train_lbl_np, test_np, test_lbl_np = load_UCR(self.dataset_name, base_data_dir=self.data_dir)
            # ts2vec returns (N, T, 1); convert to (N, 1, T)
            x_train = torch.from_numpy(train_np).float()
            x_test = torch.from_numpy(test_np).float()
            y_train = torch.from_numpy(train_lbl_np).long()
            y_test = torch.from_numpy(test_lbl_np).long()

            # Permute if needed
            if self.permute_indexes:
                # Permute from (N, T, 1) to (N, 1, T)
                x_train = x_train.permute(0, 2, 1)
                x_test = x_test.permute(0, 2, 1)
            
            # train/val split
            full = TensorDataset(x_train, y_train)
            n_q = int(len(full) * self.q_split)
            n_tr = len(full) - n_q
            self.train_dataset, self.q_dataset = random_split(
                full,
                [n_tr, n_q],
                generator=torch.Generator().manual_seed(self.seed)
            )
            # test
            self.test_dataset = TensorDataset(x_test, y_test)
            
            # --- now cache exactly those six splits ---
            train_idx = self.train_dataset.indices
            q_idx = self.q_dataset.indices

            x_train_cache = x_train[train_idx]
            y_train_cache = y_train[train_idx]
            x_q_cache = x_train[q_idx]
            y_q_cache = y_train[q_idx]
            
            # x_test / y_test are already full test set

            torch.save({
                'x_train': x_train_cache,
                'y_train': y_train_cache,
                'x_q': x_q_cache,
                'y_q': y_q_cache,
                'x_test': x_test,
                'y_test': y_test
            }, self.cache_path)

        # Set input features and number of classes for TS-MoCo compatibility
        # Get sample from train dataset to determine input features
        sample_x, _ = self.train_dataset[0]
        if len(sample_x.shape) == 3:  # (1, T) format
            self.input_features = sample_x.shape[0]  # Number of features
        else:  # (T,) format
            self.input_features = 1
        
        # Get number of classes from unique labels
        all_labels = []
        for i in range(len(self.train_dataset)):
            _, label = self.train_dataset[i]
            all_labels.append(label.item())
        self.n_classes = len(set(all_labels))
        self.class_names = [f"Class_{i}" for i in range(self.n_classes)]

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
                prefetch_factor=self.prefetch_factor
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
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
                prefetch_factor=self.prefetch_factor
            )
        else:
            return DataLoader(
                self.q_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
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
                prefetch_factor=self.prefetch_factor
            )
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
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
