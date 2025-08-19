import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

def convert_ucr_to_pt_format(ucr_data_path, output_path, dataset_name):
    """
    Convert UCR dataset files to the expected .pt format for the Lightning module.
    
    Args:
        ucr_data_path: Path to the UCR dataset files
        output_path: Path where to save the converted .pt files
        dataset_name: Name of the UCR dataset (e.g., 'Coffee', 'ECG200', etc.)
    """
    
    # UCR file naming convention
    train_file = os.path.join(ucr_data_path, f"{dataset_name}_TRAIN.txt")
    test_file = os.path.join(ucr_data_path, f"{dataset_name}_TEST.txt")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(f"UCR files not found: {train_file} or {test_file}")
    
    # Load training data
    train_data = pd.read_csv(train_file, header=None, sep='\t')
    test_data = pd.read_csv(test_file, header=None, sep='\t')
    
    # First column is the label, rest are the time series
    X_train = train_data.iloc[:, 1:].values.astype(np.float32)
    y_train = train_data.iloc[:, 0].values.astype(np.int64)
    
    X_test = test_data.iloc[:, 1:].values.astype(np.float32)
    y_test = test_data.iloc[:, 0].values.astype(np.int64)
    
    # Convert to torch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    # Create train/val split from training data
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_tensor, y_train_tensor, test_size=0.25, random_state=42, stratify=y_train_tensor
    )
    
    # Save as .pt files
    os.makedirs(output_path, exist_ok=True)
    
    # Save train data
    torch.save({
        'samples': X_train_split,
        'labels': y_train_split
    }, os.path.join(output_path, 'train.pt'))
    
    # Save validation data
    torch.save({
        'samples': X_val_split,
        'labels': y_val_split
    }, os.path.join(output_path, 'val.pt'))
    
    # Save test data
    torch.save({
        'samples': X_test_tensor,
        'labels': y_test_tensor
    }, os.path.join(output_path, 'test.pt'))
    
    print(f"Converted {dataset_name} dataset:")
    print(f"  Train samples: {X_train_split.shape[0]}")
    print(f"  Val samples: {X_val_split.shape[0]}")
    print(f"  Test samples: {X_test_tensor.shape[0]}")
    print(f"  Sequence length: {X_train_split.shape[1]}")
    print(f"  Number of classes: {len(torch.unique(y_train_tensor))}")
    print(f"  Classes: {torch.unique(y_train_tensor).tolist()}")
    print(f"  Saved to: {output_path}")

def convert_multiple_ucr_datasets(ucr_base_path, output_base_path, dataset_names):
    """
    Convert multiple UCR datasets at once.
    
    Args:
        ucr_base_path: Base path containing UCR dataset folders
        output_base_path: Base path where to save converted datasets
        dataset_names: List of dataset names to convert
    """
    for dataset_name in dataset_names:
        try:
            ucr_data_path = os.path.join(ucr_base_path, dataset_name)
            output_path = os.path.join(output_base_path, dataset_name)
            convert_ucr_to_pt_format(ucr_data_path, output_path, dataset_name)
        except Exception as e:
            print(f"Error converting {dataset_name}: {e}")

if __name__ == "__main__":
    # Example usage
    ucr_base_path = "../Datasets/UCR"  # Path to your UCR datasets
    output_base_path = "../Datasets/UCR_processed"  # Where to save converted files
    
    # List of UCR datasets you want to convert
    dataset_names = [
        "Coffee",
        "ECG200", 
        "ECGFiveDays",
        "GunPoint",
        "Lightning2",
        "Lightning7",
        "MoteStrain",
        "OliveOil",
        "OSULeaf",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
        "SwedishLeaf",
        "SyntheticControl",
        "Trace",
        "TwoLeadECG",
        "Wafer",
        "Yoga"
    ]
    
    # Convert all datasets
    convert_multiple_ucr_datasets(ucr_base_path, output_base_path, dataset_names)
    
    # Or convert a single dataset
    # convert_ucr_to_pt_format(
    #     os.path.join(ucr_base_path, "Coffee"),
    #     os.path.join(output_base_path, "Coffee"),
    #     "Coffee"
    # )
