import torch
from pathlib import Path
import argparse
import os
import sys

# Add the root directory to Python path
root_dir = str(Path(__file__).resolve().parents[3])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Project_2.src.utils.dataset_spectogram import SpeechCommandsSpectrogramDataset, read_list, create_data_loaders
from Project_2.src.CNN.models import AudioCNN
from Project_2.src.CNN.train_definition import train_model, evaluate_model

def main():
    parser = argparse.ArgumentParser(description='Train a speech command recognition model')
    parser.add_argument('--data_dir', type=str, default='Project_2/data/train', help='Path to the data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--n_mels', type=int, default=64, help='Number of mel bands')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT size')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length')
    parser.add_argument('--standardize', type=bool, default=False, help='Standardize spectrograms')
    parser.add_argument('--model_path', type=str, default='Project_2/models/CNN/audio_cnn3.pt', help='Path to save model')
    parser.add_argument('--stats_path', type=str, default='training_stats.csv', help='Path to save training stats')
    
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
        num_epochs=args.epochs,
        learning_rate=args.lr,
        file_path=args.stats_path
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...\n")
    evaluate_model(model, test_loader, train_loader)
    
    # Save model
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

if __name__ == "__main__":
    main()
