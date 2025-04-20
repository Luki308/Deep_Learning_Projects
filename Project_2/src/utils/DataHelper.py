import os
import torch
# import intel_extension_for_pytorch as ipex
from pathlib import Path
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F


def collate_fn(batch):
    """
    Pads waveforms in the batch to the longest waveform length.
    """
    waveforms, labels, sample_rates = zip(*batch)

    # Find max length
    max_length = max(waveform.shape[1] for waveform in waveforms)

    # Pad all waveforms
    padded_waveforms = []
    for waveform in waveforms:
        pad_amount = max_length - waveform.shape[1]
        padded_waveform = F.pad(waveform, (0, pad_amount))  # Pad last dimension (time)
        padded_waveforms.append(padded_waveform)

    # Stack into tensors
    waveforms_tensor = torch.stack(padded_waveforms)
    sample_rates_tensor = torch.tensor(sample_rates)  # optional, or just pass along
    return waveforms_tensor, labels, sample_rates_tensor

def read_list(file_path):
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f.readlines())
    
class SpeechCommandsDataset(Dataset):
    """
    Custom dataset for loading speech command audio files.

    Args:
        file_list (list): List of paths to audio files.
        transform (callable, optional): Optional transform to be applied on a sample. 
    """
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(str(path))
        label = path.parent.name  # Folder name is the label
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label, sample_rate

def get_data_loaders(data_dir='Project_2/data/train', batch_size=32, shuffle_train=True):
    """
    Load speech command datasets and return data loaders
    
    Args:
        data_dir: Root directory for the dataset
        batch_size: Batch size for the data loaders
        shuffle_train: Whether to shuffle the training data
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Make sure we use absolute paths
    data_dir = Path(data_dir).resolve()
    
    # Read which files belong to which set
    val_list = read_list(data_dir / 'validation_list.txt')
    test_list = read_list(data_dir / 'testing_list.txt')

    # Create a dictionary to hold the file paths for each set
    dataset_path = data_dir / 'audio'
    all_data = {'training': [], 'validation': [], 'testing': []}

    # Check if the directory exists
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Print dataset_path for debugging
    print(f"Looking for audio files in: {dataset_path}")

    for audio_path in dataset_path.rglob("*.wav"):
        relative_path = audio_path.relative_to(dataset_path).as_posix()

        if relative_path in val_list:
            all_data['validation'].append(audio_path)
        elif relative_path in test_list:
            all_data['testing'].append(audio_path)
        else:
            all_data['training'].append(audio_path)

    # Print counts for debugging
    print(f"Found {len(all_data['training'])} training files")
    print(f"Found {len(all_data['validation'])} validation files")
    print(f"Found {len(all_data['testing'])} testing files")

    print(f"{all_data['training'][:2]}")
    # Create datasets and dataloaders
    train_dataset = SpeechCommandsDataset(all_data['training'])
    val_dataset = SpeechCommandsDataset(all_data['validation'])
    test_dataset = SpeechCommandsDataset(all_data['testing'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

# Example usage:
if __name__ == "__main__":
    # Instead of trying to set the backend, ensure necessary dependencies are installed
    print(f"Available torchaudio backends: {torchaudio.list_audio_backends()}")
    
    # Test if you can load an audio file
    print("Attempting to load data...")
    train_loader, val_loader, test_loader = get_data_loaders()

    # Print example training data
    for batch in train_loader:
        waveforms, labels, sample_rates = batch
        print(f"Waveform shape: {waveforms.shape}, Labels: {labels}")
        break  # Just show one batch for demonstration