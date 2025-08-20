# TS-MoCo Integration

This document explains how to use the integrated TS-MoCo (Time Series Momentum Contrast) model in the ICLR25 repository.

## Overview

TS-MoCo has been integrated to use the global data loading infrastructure from the parent repository. The integration includes:

- **Global Data Loading**: Uses the same `data_loader` and `UCRDataModule` as other models
- **Consistent Data Paths**: Points to the shared `data/` directory
- **Wrapper Script**: Easy-to-use command-line interface

## Quick Start

### 1. Test the Integration

First, test that the integration works:

```bash
cd /Users/jessekroll/Desktop/ICLR25
python test_tsmoco_integration.py
```

### 2. Run TS-MoCo

Use the wrapper script to run TS-MoCo with your desired dataset:

```bash
cd /Users/jessekroll/Desktop/ICLR25
python run_tsmoco.py Coffee 0 128 2 2 1 6 1e-4 0.9 1.0 0.5 temporal_window_masking 10 0 1 random standardize
```

### 3. Direct Command (Alternative)

You can also run TS-MoCo directly:

```bash
cd /Users/jessekroll/Desktop/ICLR25/models/tsmoco
python main_cli.py UCR 0 128 2 2 1 6 1e-4 0.9 1.0 0.5 temporal_window_masking 10 0 1 random standardize
```

## Command Line Arguments

The TS-MoCo command takes the following arguments:

```
python main_cli.py UCR <pos_embeddings_alpha> <embedding_dim> <n_head_token_enc> <n_head_context_enc> <depth_context_enc> <max_predict_len> <lr> <tau> <lam> <masking_percentage> <masking_method> <pretrain_epochs> <finetune_epochs> <es_after_epochs> <train_val_split> <preprocessing>
```

### Parameter Descriptions

- `pos_embeddings_alpha`: Multiplication factor for positional embedding vector
- `embedding_dim`: Number of features in latent representation
- `n_head_token_enc`: Number of heads in token encoder
- `n_head_context_enc`: Number of heads in context encoder
- `depth_context_enc`: Depth of context encoder
- `max_predict_len`: Maximum future timesteps to predict
- `lr`: Learning rate for optimization
- `tau`: Momentum value
- `lam`: Loss weight
- `masking_percentage`: Percentage of input to mask (0.0-1.0)
- `masking_method`: Masking algorithm (`random`, `channel_wise`, `temporal`, `temporal_window_masking`)
- `pretrain_epochs`: Number of epochs for self-supervised pretraining
- `finetune_epochs`: Number of epochs for supervised fine-tuning
- `es_after_epochs`: Early stopping patience
- `train_val_split`: Validation split method (`random`, `subject`)
- `preprocessing`: Data preprocessing (`None`, `standardize`, `normalize`)

## Example Commands

### Basic Training
```bash
python run_tsmoco.py Coffee 0 128 2 2 1 6 1e-4 0.9 1.0 0.5 temporal_window_masking 10 0 1 random standardize
```

### Quick Test (1 epoch each)
```bash
python run_tsmoco.py Coffee 0 64 2 2 1 4 1e-3 0.9 1.0 0.3 temporal_window_masking 1 1 1 random standardize
```

### Different Dataset
```bash
python run_tsmoco.py Adiac 0 128 2 2 1 6 1e-4 0.9 1.0 0.5 temporal_window_masking 10 0 1 random standardize
```

## Configuration

### Data Path
The data path is configured in `models/tsmoco/device_hyperparameters.json`:
```json
{
  "ss_ucr_datapath": "../../data",
  "ss_ucr_dataset_name": "Coffee",
  "ss_ucr_batch_size": 16
}
```

### Dataset Name
The dataset name is automatically updated by the wrapper script based on the command line argument.

## Output

TS-MoCo will create the following outputs in `models/tsmoco/logs/`:

- **Checkpoints**: Model checkpoints for pretraining and fine-tuning
- **CSV Logs**: Training metrics and loss curves
- **TensorBoard Logs**: Detailed training visualizations
- **Embeddings**: Extracted embeddings for train/val/test sets

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the correct directory
2. **Data Not Found**: Verify the dataset exists in `data/UCR/<dataset_name>/`
3. **CUDA Issues**: Check GPU availability and PyTorch installation

### Environment Setup

TS-MoCo requires the following dependencies (see `models/tsmoco/env.yaml`):
- Python 3.10
- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- Other dependencies listed in the env.yaml file

## Integration Details

### Data Loading Changes

The integration replaces the local UCR data loading with the global data loading:

- **Before**: Used `UCR_preprocess.py` for local data loading
- **After**: Uses `data_loader` from the parent repository

### Key Files Modified

1. `models/tsmoco/datasets/ucr_dataset.py`: Updated to use global data loading
2. `models/tsmoco/device_hyperparameters.json`: Updated data paths
3. `run_tsmoco.py`: New wrapper script for easy execution
4. `test_tsmoco_integration.py`: Test script for verification

### Data Format

The integration handles data format conversion:
- **Input**: Global data loader provides (N, T, C) format
- **TS-MoCo**: Expects (N, C, T) format when `permute_indexes=True`
- **Conversion**: Automatic permutation in the datamodule
