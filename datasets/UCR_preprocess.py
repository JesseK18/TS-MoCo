import os
import glob
import numpy as np
import pandas as pd
import pickle
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def _find_ucr_pair(base_dir: str, dataset_name: str):
    """Find TRAIN/TEST files for a UCR dataset, supporting .tsv/.txt/.ts and any whitespace-delimited format."""
    preferred_exts = ["tsv", "txt", "ts", "TSV", "TXT", "TS"]
    # Try preferred list first
    for ext in preferred_exts:
        train_candidate = os.path.join(base_dir, f"{dataset_name}_TRAIN.{ext}")
        test_candidate = os.path.join(base_dir, f"{dataset_name}_TEST.{ext}")
        if os.path.exists(train_candidate) and os.path.exists(test_candidate):
            return train_candidate, test_candidate
    # Fallback: any extension
    any_train = sorted(glob.glob(os.path.join(base_dir, f"{dataset_name}_TRAIN.*")))
    any_test = sorted(glob.glob(os.path.join(base_dir, f"{dataset_name}_TEST.*")))
    if any_train and any_test:
        return any_train[0], any_test[0]
    return None, None

def load_UCR(dataset_name, base_data_dir=None):
    """
    Load a UCR dataset.
    If base_data_dir is provided, expect files under base_data_dir/<dataset_name>/.
    Otherwise, default to <repo_root>/data/UCR/<dataset_name>/.
    """
    # Determine base directory for dataset files
    if base_data_dir is None:
        root_dir = os.path.dirname(os.path.dirname(__file__))
        base_dir = os.path.join(root_dir, 'data', dataset_name)
    else:
        base_dir = os.path.join(base_data_dir, dataset_name)
    train_file, test_file = _find_ucr_pair(base_dir, dataset_name)

    # Validate that the files exist
    if not train_file or not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Train file not found for dataset '{dataset_name}'. Expected under: {base_dir}\n"
            f"Place files as '{dataset_name}_TRAIN.tsv' and '{dataset_name}_TEST.tsv' (or .txt/.ts)."
        )
    if not test_file or not os.path.exists(test_file):
        raise FileNotFoundError(
            f"Test file not found for dataset '{dataset_name}'. Expected under: {base_dir}\n"
            f"Place files as '{dataset_name}_TRAIN.tsv' and '{dataset_name}_TEST.tsv' (or .txt/.ts)."
        )
    
    # Use whitespace regex to handle TSV, TXT, TS uniformly
    train_df = pd.read_csv(train_file, sep=r'\s+', header=None, engine='python')
    test_df = pd.read_csv(test_file, sep=r'\s+', header=None, engine='python')
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Handle NaN values
    if np.isnan(train).any():
        train = np.nan_to_num(train, nan=0.0)
    if np.isnan(test).any():
        test = np.nan_to_num(test, nan=0.0)
    
    # Standardization
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
