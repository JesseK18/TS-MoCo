#!/usr/bin/env python3
"""
Wrapper script to run TS-MoCo with correct environment setup and data paths.
Usage: python run_tsmoco.py <dataset_name> [additional_args...]
Example: python run_tsmoco.py Coffee 0 128 2 2 1 6 1e-4 0.9 1.0 0.5 temporal_window_masking 10 0 1 random standardize
"""

import os
import sys
import subprocess
import argparse

def main():
    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # Change to the TS-MoCo directory
    tsmoco_dir = os.path.join(current_dir, 'models', 'tsmoco')
    os.chdir(tsmoco_dir)
    
    # Get command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_tsmoco.py <dataset_name> [additional_args...]")
        print("Example: python run_tsmoco.py Coffee 0 128 2 2 1 6 1e-4 0.9 1.0 0.5 temporal_window_masking 10 0 1 random standardize")
        sys.exit(1)
    
    # Extract dataset name and additional arguments
    dataset_name = sys.argv[1]
    additional_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # Update the device hyperparameters with the dataset name
    import json
    device_params_path = os.path.join(tsmoco_dir, 'device_hyperparameters.json')
    
    with open(device_params_path, 'r') as f:
        device_params = json.load(f)
    
    device_params['ss_ucr_dataset_name'] = dataset_name
    
    with open(device_params_path, 'w') as f:
        json.dump(device_params, f, indent=2)
    
    # Build the command
    cmd = ['python', 'main_cli.py', 'UCR'] + additional_args
    
    print(f"Running TS-MoCo with dataset: {dataset_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("TS-MoCo completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"TS-MoCo failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("TS-MoCo interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
