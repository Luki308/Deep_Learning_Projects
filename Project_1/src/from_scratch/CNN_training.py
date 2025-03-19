# Create CNN to classify CINIC-10 datase using PyTorch and tutorial from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Command to run: tensorboard --logdir=Project_1/runs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(0)

import numpy as np
import itertools
from tqdm import tqdm
import os
from datetime import datetime
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from Project_1.src.DataHelper import DataHelper
from Project_1.src.from_scratch.CNN_definiton import CNN_ours

# Define hyperparameters as a dictionary for easy logging
config = {
    'model': 'BASIC',
    'lr': 0.01,  # List of learning rates to try
    'batch_size': 32,  # List of batch sizes to try
    'epochs': 150,
    'early_stopping_patience': 5,
    'layers': 'basic',
    'subset_percentage': 0.5,
}

def get_run_name(config):
    """Create a unique name for each experiment run"""
    timestamp = datetime.now().strftime("%d.%m-%H.%M")
    return f"{config['layers']}-layers_lr{config['lr']}_batch{config['batch_size']}_ep{config['epochs']}_subset{int(config['subset_percentage']*100)}pc"

def get_hyperparameter_combinations(config):
    """Create all combinations of hyperparameters for grid search"""
    # Identify which hyperparameters are lists (those we want to search over)
    search_params = {}
    fixed_params = {}
    
    for key, value in config.items():
        if isinstance(value, list):
            search_params[key] = value
        else:
            fixed_params[key] = value
    
    # Generate all combinations of the search parameters
    keys = list(search_params.keys())
    values = list(search_params.values())
    combinations = list(itertools.product(*values))
    
    # Create a list of complete parameter sets
    all_configs = []
    for combo in combinations:
        combo_config = fixed_params.copy()
        for i, key in enumerate(keys):
            combo_config[key] = combo[i]
        all_configs.append(combo_config)
    
    return all_configs

def train_model(config, tuning_run_dir):
    """Train a model with the given configuration"""
    # Create a unique run name
    run_name = get_run_name(config)
    print(f"\n\n=== Training with config: lr={config['lr']}, batch_size={config['batch_size']} ===\n")
    
    # Initialize TensorBoard writer to use the tuning run directory
    writer_path = os.path.join(tuning_run_dir, run_name)
    writer = SummaryWriter(writer_path)
    
    # Create the network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CNN_ours().to(device)
    
    # # Log model graph
    # example_input = torch.rand(1, 3, 32, 32).to(device)
    # writer.add_graph(net, example_input)
    
    # Get data with the specific batch size
    DataH = DataHelper(resize=32, batch_size=config['batch_size'], subset_fraction=config['subset_percentage'])
    trainloader, valloader, testloader = DataH.get_loaders()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(net.parameters(), lr=config['lr'])
    
    # Training parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = config['early_stopping_patience']
    
    # Print every 10% of the dataset
    print_every = max(1, int(90000 * config['subset_percentage'] // config['batch_size'] // 10))
    
    # Train the network
    for epoch in tqdm(range(config['epochs'])):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_accuracies = []
        train_losses = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())

            # backward
            loss.backward()
            optimizer.step()

            # Track training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_accuracy = 100 * correct_train / total_train
            train_accuracies.append(train_accuracy)

            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every-1:  # print every x mini-batches
                print(f"\n[(Train)Epoch:{epoch + 1}, Batch {i + 1} / {len(trainloader)}] loss: {running_loss / print_every:.3f}, accuracy: {train_accuracy:.2f}%")
                running_loss = 0.0

        # Calculate mean loss and accuracy for the epoch
        mean_train_loss = np.mean(train_losses)
        mean_train_accuracy = np.mean(train_accuracies)
        
        # Log the epoch means to TensorBoard
        writer.add_scalar('Loss/train', mean_train_loss, epoch)
        writer.add_scalar('Accuracy/train', mean_train_accuracy, epoch)        

        # Validate the network
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_accuracies = []
        val_losses = []

        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                images, labels = data[0].to(device), data[1].to(device)
                # forward
                outputs = net(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_losses.append(loss.item())
                
                # check predictions
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_accuracy = 100 * correct / total
                val_accuracies.append(val_accuracy)

                running_loss += loss.item()
                if i % print_every == print_every-1:  # print every x mini-batches
                    print(f"\n[(Val)Epoch:{epoch + 1}, Batch {i + 1} / {len(trainloader)}] loss: {running_loss / print_every:.3f}, accuracy: {val_accuracy:.2f}%")
                    running_loss = 0.0
        
        # Calculate mean validation loss and accuracy
        val_loss = np.mean(val_losses)
        val_accuracy = 100 * correct / total if total > 0 else 0
        
        # Log validation means to TensorBoard
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)   

        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            break

    # Log hyperparameters and metrics to TensorBoard
    hparam_dict = config.copy()  # Start with all config parameters

    # Add metrics to the hyperparameter dictionary since this is required for visualization
    hparam_dict.update({
        'train_loss': round(float(np.mean(train_losses)), 3),
        'train_acc': round(float(np.mean(train_accuracies)), 2),
        'val_loss': round(float(val_loss), 3),
        'val_acc': round(float(val_accuracy), 2)
    })

    # For TensorBoard metrics in proper format
    metrics_dict = {
        'hparam/train_loss': round(float(np.mean(train_losses)), 3),
        'hparam/train_accuracy': round(float(np.mean(train_accuracies)), 2),
        'hparam/val_loss': round(float(val_loss), 3),
        'hparam/val_accuracy': round(float(val_accuracy), 2)
    }

    # Log using add_hparams
    writer.add_hparams(
        hparam_dict,
        metrics_dict
    )
    writer.flush()
    writer.close()
    
    # Save the model
    if 'Project_1' in os.listdir():
        if not os.path.exists(f'Project_1/models/{config["model"]}'):
            os.makedirs(f'Project_1/models/{config["model"]}')
        PATH = f'Project_1/models/{config["model"]}/{run_name+'_time-'+datetime.now().strftime("%d.%m-%H.%M")}.pth'
    else:
        if not os.path.exists(f'models/{config["model"]}'):
            os.makedirs(f'models/{config["model"]}')
        PATH = f'models/{config["model"]}/{run_name}.pth'
        
    torch.save(net.state_dict(), PATH)
    
    return {
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'train_loss': mean_train_loss,
        'train_accuracy': mean_train_accuracy,
        'config': config,
        'model_path': PATH
    }

if __name__ == "__main__":
    # Define the device to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate all hyperparameter combinations to test
    all_configs = get_hyperparameter_combinations(config)
    print(f"Generated {len(all_configs)} hyperparameter combinations to test")
    
    # Create a single directory for this tuning run
    tuning_run_timestamp = datetime.now().strftime("%d.%m-%H.%M")
    tuning_run_dir = f'Project_1/runs/{config["model"]}/tuning_{tuning_run_timestamp}'
    if not os.path.exists(tuning_run_dir):
        os.makedirs(tuning_run_dir)
    print(f"Saving all runs to: {tuning_run_dir}")

    # Train a model for each configuration
    results = []
    for i, config_instance in enumerate(all_configs):
        print(f"\nTraining model {i+1}/{len(all_configs)}")
        result = train_model(config_instance)
        results.append(result)

    # Find best model
    best_model = min(results, key=lambda x: x['val_loss'])
    print("\n\n=== Best Model ===")
    print(f"Config: {best_model['config']}")
    print(f"Validation Loss: {best_model['val_loss']:.4f}")
    print(f"Validation Accuracy: {best_model['val_accuracy']:.2f}%")
    print(f"Model saved at: {best_model['model_path']}")