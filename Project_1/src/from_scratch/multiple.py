import numpy as np
import json
import os
import torch
torch.manual_seed(0)
from datetime import datetime
from tqdm import tqdm
import sys
import copy

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from Project_1.src.from_scratch.CNN_training import train_model, get_run_name

def run_multiple_trainings(config, num_runs=5):
    """
    Run multiple trainings with the same configuration to calculate statistics.
    
    Args:
        config: Dictionary containing model configuration
        num_runs: Number of times to run the training
    
    Returns:
        Dictionary with statistics about the runs
    """
    print(f"Running {num_runs} trainings with the same configuration")
    print(f"Configuration: {config}")
    
    # Create directory for this statistical run
    timestamp = datetime.now().strftime("%d.%m-%H.%M")
    stats_dir = f'Project_1/stats/{"_".join([f"{k}_{v}" for k, v in config.items() if k in ["model", "lr", "batch_size", "subset_percentage"]])}'
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    
    # Define the device to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create directory for this run's TensorBoard logs
    global tuning_run_dir
    tuning_run_dir = f'Project_1/runs/{config["model"]}/stats_run_{timestamp}'
    if not os.path.exists(tuning_run_dir):
        os.makedirs(tuning_run_dir)
    
    # Run training multiple times
    results = []
    for i in tqdm(range(num_runs), desc="Training runs"):
        # Create a copy of the config to avoid modifying the original
        run_config = copy.deepcopy(config)
        
        # Train the model and collect results
        print(f"\nRun {i+1}/{num_runs}")
        result = train_model(run_config, tuning_run_dir=tuning_run_dir)
        results.append(result)
    
    # Calculate statistics
    train_losses = [result['train_loss'] for result in results]
    train_accuracies = [result['train_accuracy'] for result in results]

    val_losses = [result['val_loss'] for result in results]
    val_accuracies = [result['val_accuracy'] for result in results]

    
    stats = {
        'config': config,
        'num_runs': num_runs,
        'train_loss_mean': float(np.mean(train_losses)),
        'train_loss_std': float(np.std(train_losses)),
        'train_accuracy_mean': float(np.mean(train_accuracies)),
        'train_accuracy_std': float(np.std(train_accuracies)),
        'val_loss_mean': float(np.mean(val_losses)),
        'val_loss_std': float(np.std(val_losses)),
        'val_accuracy_mean': float(np.mean(val_accuracies)),
        'val_accuracy_std': float(np.std(val_accuracies)),
        'individual_results': [
            {
                'train_loss': float(result['train_loss']),
                'train_accuracy': float(result['train_accuracy']),
                'val_loss': float(result['val_loss']), 
                'val_accuracy': float(result['val_accuracy']),
                'model_path': result['model_path']
            } for result in results
        ]
    }
    
    # Save results to JSON file
    stats_file = os.path.join(stats_dir, f'stats_{timestamp}.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Print summary
    print("\n===== Training Statistics =====")
    print(f"Configuration: {config}")
    print(f"Number of runs: {num_runs}")
    print(f"Validation Loss: {stats['val_loss_mean']:.4f} ± {stats['val_loss_std']:.4f}")
    print(f"Validation Accuracy: {stats['val_accuracy_mean']:.2f}% ± {stats['val_accuracy_std']:.2f}%")
    print(f"Statistics saved to: {stats_file}")
    
    return stats

if __name__ == "__main__":
    # Define the configuration to test
    num_runs = 5
    config = {
        'model': 'BASIC',
        'lr': 0.0001,
        'batch_size': 256,
        'epochs': 50,
        'early_stopping_patience': 5,
        'layers': 'basic',
        'subset_percentage': 0.2,
    }

    # Run the statistical analysis
    stats = run_multiple_trainings(config, num_runs)


    config = {
        'model': 'BASIC',
        'lr': 0.00001,
        'batch_size': 256,
        'epochs': 50,
        'early_stopping_patience': 5,
        'layers': 'basic',
        'subset_percentage': 0.2,
    }

    # Run the statistical analysis
    stats = run_multiple_trainings(config, num_runs)