import os
from pathlib import Path
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
        waveform, sample_rate = torchaudio.load(path)
        label = path.parent.name  # Folder name is the label
        if self.transform:
            waveform = self.transform(waveform)
        return waveform, label

def get_data_loaders(data_dir='data/train', batch_size=32, shuffle_train=True):
    """
    Load speech command datasets and return data loaders
    
    Args:
        data_dir: Root directory for the dataset
        batch_size: Batch size for the data loaders
        shuffle_train: Whether to shuffle the training data
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Read which files belong to which set
    val_list = read_list(f'{data_dir}/validation_list.txt')
    test_list = read_list(f'{data_dir}/testing_list.txt')

    # Create a dictionary to hold the file paths for each set
    dataset_path = Path(f'{data_dir}/audio')
    all_data = {'training': [], 'validation': [], 'testing': []}

    for audio_path in dataset_path.rglob("*.wav"):
        relative_path = audio_path.relative_to(dataset_path).as_posix()

        if relative_path in val_list:
            all_data['validation'].append(audio_path)
        elif relative_path in test_list:
            all_data['testing'].append(audio_path)
        else:
            all_data['training'].append(audio_path)

    # Create datasets and dataloaders
    train_dataset = SpeechCommandsDataset(all_data['training'])
    val_dataset = SpeechCommandsDataset(all_data['validation'])
    test_dataset = SpeechCommandsDataset(all_data['testing'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

# Example usage:
# train_loader, val_loader, test_loader = get_data_loaders()