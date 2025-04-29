import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse
import os
import sys

# Add the root directory to Python path
root_dir = str(Path(__file__).resolve().parents[3])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Project_2.src.utils.dataset_spectogram import SpeechCommandsSpectrogramDataset, read_list, create_data_loaders, collate_spectrograms
from Project_2.src.CNN.models import AudioCNN
from Project_2.src.CNN.train_definition import train_model, evaluate_model

def main(parser:argparse.ArgumentParser, which_iteration:int=1):
    
    args = parser.parse_args()
    
    # Ensure paths exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Set up data
    data_dir = Path(args.data_dir).resolve()
    
    # Read which files belong to which set
    val_list = read_list(data_dir / 'validation_list.txt')
    test_list = read_list(data_dir / 'testing_list.txt')
    
    # Create a dictionary to hold the file paths for each set
    dataset_path = data_dir / 'audio'
    all_data = {'training': [], 'validation': [], 'testing': []}
    
    # Check if the directory exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    print(f"Looking for audio files in: {dataset_path}")
    
    for audio_path in dataset_path.rglob("*.wav"):
        relative_path = audio_path.relative_to(dataset_path).as_posix()
        
        if relative_path in val_list:
            all_data['validation'].append(audio_path)
        elif relative_path in test_list:
            all_data['testing'].append(audio_path)
        else:
            all_data['training'].append(audio_path)
    
    # Create datasets
    train_dataset = SpeechCommandsSpectrogramDataset(
        all_data['training'], 
        n_mels=args.n_mels, 
        n_fft=args.n_fft, 
        hop_length=args.hop_length, 
        standardize=args.standardize
    )
    
    val_dataset = SpeechCommandsSpectrogramDataset(
        all_data['validation'], 
        n_mels=args.n_mels, 
        n_fft=args.n_fft, 
        hop_length=args.hop_length, 
        standardize=False
    )
    
    test_dataset = SpeechCommandsSpectrogramDataset(
        all_data['testing'], 
        n_mels=args.n_mels, 
        n_fft=args.n_fft, 
        hop_length=args.hop_length, 
        standardize=False
    )
    
    # If standardization is enabled, apply training stats to validation and test sets
    if args.standardize:
        val_dataset.standardize = True
        val_dataset.spec_mean = train_dataset.spec_mean
        val_dataset.spec_std = train_dataset.spec_std
        
        test_dataset.standardize = True
        test_dataset.spec_mean = train_dataset.spec_mean
        test_dataset.spec_std = train_dataset.spec_std
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=args.batch_size
    )
    
    # Create model
    model = AudioCNN(num_classes=train_dataset.num_classes, n_mels=args.n_mels)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        which_iteration=which_iteration,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        file_path=args.stats_path
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...\n")
    evaluate_model(model, test_loader, train_loader, args.stats_path, which_iteration = which_iteration)
    
    # Save model
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

def predict_on_test_set(num_classes=12):
    parser = argparse.ArgumentParser(description='Train a speech command recognition model')
    parser.add_argument('--data_dir', type=str, default='Project_2/data/test', help='Path to the data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--n_mels', type=int, default=64, help='Number of mel bands')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT size')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length')
    parser.add_argument('--standardize', type=bool, default=False, help='Standardize spectrograms')
    parser.add_argument('--model_path', type=str, default='Project_2/models/CNN/audio_cnn_slc2.pt', help='Path to save model')
    parser.add_argument('--stats_path', type=str, default='Project_2/results/CNN/test_stats_CNN_slc2.csv', help='Path to save training stats')
    parser.add_argument('--plot_path', type=str, default='Project_2/results/CNN/class_distribution_slc2.png', help='Path to save class distribution plot')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Check if the model path exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    else:
        print(f"Model path exists: {args.model_path}")
  
    # Set up data
    data_dir = Path(args.data_dir).resolve()
    
    # Create a list to hold all test files
    dataset_path = data_dir / 'audio'
    all_data = []
    
    # Check if the directory exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    print(f"Looking for audio files in: {dataset_path}")
    
    for audio_path in dataset_path.rglob("*.wav"):
        all_data.append(audio_path)
    
    # Create test dataset
    test_dataset = SpeechCommandsSpectrogramDataset(
        all_data, 
        n_mels=args.n_mels, 
        n_fft=args.n_fft, 
        hop_length=args.hop_length, 
        standardize=False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_spectrograms,
    )
        
    # Load the model
    model = AudioCNN(num_classes=num_classes, n_mels=args.n_mels)
    model_params = torch.load(args.model_path)
    model.load_state_dict(model_params)
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Initialize counts for each class
    class_counts = [0] * num_classes
    all_predictions = []

    # Predict on test set
    print("\nPredicting on test set...\n")
    for batch in tqdm(test_loader, desc="Predicting", unit="batch"):
        specs, _, file_paths = batch  # Ignore labels as they don't exist
        specs = specs.to(device)
        
        with torch.no_grad():
            outputs = model(specs)
            _, predicted = outputs.max(1)
            
            # Collect predictions
            all_predictions.extend(predicted.cpu().numpy())
            
            # Update class counts
            for i in range(num_classes):
                class_counts[i] += (predicted == i).sum().item()
    
    # Print number of samples in each class
    print("\nPredicted class distribution:")
    for i in range(num_classes):
        print(f"Class {i}: {class_counts[i]} samples")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(args.plot_path), exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        class_indices = np.arange(num_classes)
        
        plt.bar(class_indices, class_counts)
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.title('Predicted Class Distribution')
        # plt.xticks(class_indices)
        # X ticks are set to class names in the dataset
        plt.xticks(class_indices, test_dataset.labels, rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add count labels on top of each bar
        for i, count in enumerate(class_counts):
            plt.text(i, count + 0.5, str(count), ha='center')
        
        plt.tight_layout()
        plt.savefig(args.plot_path)
        print(f"Class distribution plot saved to {args.plot_path}")
        plt.close()

if __name__ == "__main__":
    for i in range(3, 5):
        print(f"Running experiment {i}...")
        # Set up the argument parser
        parser = argparse.ArgumentParser(description='Train a speech command recognition model')
        parser.add_argument('--data_dir', type=str, default='Project_2/data/train', help='Path to the data directory')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
        parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
        parser.add_argument('--n_mels', type=int, default=64, help='Number of mel bands')
        parser.add_argument('--n_fft', type=int, default=1024, help='FFT size')
        parser.add_argument('--hop_length', type=int, default=512, help='Hop length')
        parser.add_argument('--standardize', type=bool, default=False, help='Standardize spectrograms')
        parser.add_argument('--model_path', type=str, default=f'Project_2/models/CNN/audio_cnn_slc{i}.pt', help='Path to save model')
        parser.add_argument('--stats_path', type=str, default=f'Project_2/results/CNN/training_stats_CNN_slc{i}.csv', help='Path to save training stats')
        main(parser, which_iteration=i)

    # OR

    # predict_on_test_set()