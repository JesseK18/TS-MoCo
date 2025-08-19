# UCR Dataset Integration Guide

This guide explains how to integrate UCR (University of California, Riverside) time series datasets into your TS-MoCo repository using your existing UCR data loading pipeline.

## Overview

The UCR dataset integration uses your existing well-structured UCR data loading pipeline:
- **DataModule**: `UCRDataModule` in `datasets/ucr_dataset.py` (adapted from your existing pipeline)
- **Preprocessing**: `UCR_preprocess.py` with your existing `load_UCR` function
- **Integration Points**: Updated `main.py`, `main_supervised.py`, and `main_cli.py`
- **Configuration**: Added UCR-specific parameters to `device_hyperparameters.json`

## Key Integration Endpoints

### 1. Data Flow Architecture
```
UCR Data (.tsv files) → UCR_preprocess.py → UCRDataModule → TSMC Encoder → DenseClassifier
```

### 2. Main Integration Points
- **`main.py`** (lines 40-80): Self-supervised pretraining with UCR data
- **`main_supervised.py`** (lines 40-80): Supervised fine-tuning with UCR data  
- **`device_hyperparameters.json`**: Configuration for UCR data paths and dataset names
- **`datasets/ucr_dataset.py`**: Lightning DataModule (adapted from your existing pipeline)
- **`datasets/UCR_preprocess.py`**: Your existing preprocessing logic

## Setup Instructions

### Step 1: Prepare Your UCR Data

1. **Download UCR datasets** from the official repository
2. **Organize your data** in the expected structure:

```
/path/to/UCR/datasets/
├── Coffee/
│   ├── Coffee_TRAIN.tsv
│   └── Coffee_TEST.tsv
├── ECG200/
│   ├── ECG200_TRAIN.tsv
│   └── ECG200_TEST.tsv
└── ...
```

### Step 2: Update Configuration

Add UCR parameters to your `device_hyperparameters.json`:

```json
{
  "ss_ucr_datapath": "/path/to/your/UCR/datasets",
  "ss_ucr_dataset_name": "Coffee",
  "ss_ucr_batch_size": 64
}
```

### Step 3: Run Training

#### Self-supervised pretraining + supervised fine-tuning:
```bash
python main_cli.py UCR 0.1 128 4 4 2 6 1e-4 0.9 1.0 0.5 temporal_window_masking 100 50 10 random standardize
```

#### Supervised training only:
```bash
python main_supervised.py --dataset UCR --pos_embeddings_alpha 0.1 --embedding_dim 128 --n_head_token_enc 4 --n_head_context_enc 4 --depth_context_enc 2 --max_predict_len 0 --lr 1e-4 --finetune_epochs 50 --es_after_epochs 10 --train_val_split random --preprocessing standardize
```

## Data Format Requirements

### Input Format
Your UCR data should be organized as:
```
/path/to/UCR/datasets/
├── Coffee/
│   ├── Coffee_TRAIN.tsv  # Tab-separated: label, time_series_data
│   └── Coffee_TEST.tsv
├── ECG200/
│   ├── ECG200_TRAIN.tsv
│   └── ECG200_TEST.tsv
└── ...
```

### .tsv File Structure
Each `.tsv` file should contain:
- First column: class labels
- Remaining columns: time series data (tab-separated)

## Key Features from Your Pipeline

### 1. Caching System
- Automatic caching of processed data in `data/UCR_cache/`
- Speeds up subsequent runs by avoiding reprocessing

### 2. Train/Validation Split
- Uses your `q_split` parameter (default 0.15) for validation set
- Random split with fixed seed for reproducibility

### 3. Data Preprocessing
- Automatic standardization (mean=0, std=1)
- NaN handling with zero-filling
- Label remapping to {0, ..., L-1}

### 4. Permutation Support
- Optional `permute_indexes` parameter for data augmentation
- Converts from (N, T, 1) to (N, 1, T) format

## Using Your Existing Data Loader

You can also use your existing data loader function:

```python
from utils.ucr_data_loader import data_loader

config = {
    'data_dir': '/path/to/UCR/datasets',
    'dataset': 'Coffee',
    'batch_size': 64,
    'num_workers': 4,
    'q_split': 0.15,
    'seed': 18,
    'permute_indexes': False
}

datamodule = data_loader(config)
```

## Custom Early Stopping

Use the custom early stopping that monitors train accuracy:

```python
from utils.custom_early_stopping import CustomEarlyStopping

early_stopping = CustomEarlyStopping(
    patience=7,
    min_delta=0,
    monitor='train_acc'
)
```

## Testing the Integration

Test your UCR dataset setup:

```python
from datasets.ucr_dataset import UCRDataModule

# Test with a specific UCR dataset
datamodule = UCRDataModule(
    data_dir="/path/to/UCR/datasets",
    dataset_name="Coffee",
    batch_size=64,
    num_workers=1,
    q_split=0.15,
    seed=18,
    permute_indexes=False
)

datamodule.prepare_data()
datamodule.setup()

trainloader = datamodule.train_dataloader()
valloader = datamodule.val_dataloader()
testloader = datamodule.test_dataloader()

for x, y in trainloader:
    print(f"Input shape: {x.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Number of classes: {datamodule.n_classes}")
    print(f"Input features: {datamodule.input_features}")
    break
```

## Troubleshooting

### Common Issues

1. **File not found errors**: Check that your UCR data paths are correct in `device_hyperparameters.json`

2. **Import errors**: Make sure `datasets/UCR_preprocess.py` is available

3. **Memory issues**: Reduce batch size in `device_hyperparameters.json`

4. **Cache issues**: Clear the cache directory `data/UCR_cache/` if needed

### Data Validation

Use the test script in `datasets/ucr_dataset.py`:
```bash
cd datasets
python ucr_dataset.py
```

## Integration with Existing Workflow

The UCR integration seamlessly fits into your existing workflow:

1. **Pretraining**: Uses `plEncodingModule` with masking strategies
2. **Fine-tuning**: Uses `plClassificationModule` with frozen/unfrozen encoder
3. **Logging**: Integrates with existing CSV and TensorBoard loggers
4. **Checkpointing**: Saves best models based on validation loss
5. **Early Stopping**: Custom callback for train accuracy monitoring

## Performance Considerations

- **Caching**: First run processes data, subsequent runs use cache
- **Memory Usage**: Longer sequences may require smaller batch sizes
- **Training Time**: Consider using `limit_train_batches` for quick testing

## Next Steps

1. Place your UCR datasets in the expected directory structure
2. Update `device_hyperparameters.json` with your data paths and dataset name
3. Test with a small dataset first
4. Run full training with your preferred hyperparameters
