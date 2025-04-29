import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import numpy as np
from pathlib import Path

class SpeechCommandsSpectrogramDataset(Dataset):
    def __init__(self, file_list, n_mels=64, n_fft=1024, hop_length=512, standardize=False):
        self.file_list = file_list
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.standardize = standardize
        
        # Create the Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,  # Default sample rate for this dataset
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        
        # Create a decibel converter
        self.amp_to_db = T.AmplitudeToDB()
        
        # Define the target classes and special categories
        self.target_classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        self.labels = self.target_classes + ["unknown", "silence"]
        # Map original folder names to our target labels
        self.label_to_index = {label: i for i, label in enumerate(self.labels)}
        
        # If standardization is enabled, compute dataset statistics
        if self.standardize:
            self.compute_stats()
        
        print(f"Found {len(self.labels)} unique classes: {self.labels}")
        
    def compute_stats(self, sample_size=500):
        """Compute mean and std of mel spectrograms for standardization
        
        Standardizing the mel spectrogram values (subtracting mean and dividing by standard deviation) offers several advantages:

        1. Improved Training Stability: Neural networks generally train better with normalized inputs, as this helps with gradient flow and prevents saturation of activation functions.

        2. Faster Convergence: Standardized inputs typically lead to faster convergence during training because the optimizer doesn't need to compensate for features with different scales.

        3. Volume Invariance: Standardization helps make the model more invariant to different recording volumes, as it normalizes the intensity of the spectrograms.

        4. Better Generalization: Models trained on standardized inputs can generalize better to new data with different recording conditions.

        The implementation computes dataset statistics on a random sample of the (train) data, which is an efficient approach for large datasets. For production systems, you might want to precompute these statistics on the entire dataset or use a larger sample size.
        """
        print(f"Computing dataset statistics on {min(sample_size, len(self.file_list))} samples...")
        # Take a subset of files to compute stats (for efficiency)
        sample_files = random.sample(self.file_list, min(sample_size, len(self.file_list)))
        
        # Collect all spectrogram values
        all_values = []
        for file_path in sample_files:
            waveform, sample_rate = torchaudio.load(str(file_path))
            mel_spec = self.mel_transform(waveform)
            mel_spec_db = self.amp_to_db(mel_spec)
            all_values.append(mel_spec_db.reshape(-1))  # Flatten the spectrogram
            
        # Concatenate all values and compute statistics
        all_values = np.concatenate(all_values)
        self.spec_mean = float(np.mean(all_values))
        self.spec_std = float(np.std(all_values))
        print(f"Dataset statistics - Mean: {self.spec_mean:.2f}, Std: {self.spec_std:.2f}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        path = self.file_list[idx]
        # Load the waveform
        waveform, sample_rate = torchaudio.load(str(path))
        
        # Convert to Mel spectrogram
        mel_spec = self.mel_transform(waveform)
        # Convert to decibels (log scale) - this helps the model learn better
        mel_spec_db = self.amp_to_db(mel_spec)
        
        # Standardize if enabled
        if self.standardize:
            mel_spec_db = (mel_spec_db - self.spec_mean) / self.spec_std
        
        # Get the original label from folder name
        original_label = path.parent.name
        
        # Map to target labels based on the requirements
        if original_label in self.target_classes or original_label == "silence":
            label = original_label
        else:
            label = "unknown"
            
        label_idx = self.label_to_index[label]
        
        return mel_spec_db, label_idx, label
    
    @property
    def num_classes(self):
        return len(self.labels)


def collate_spectrograms(batch):
    # Unpack all values
    specs, label_ids, labels = zip(*batch)
    
    # Find max width (time dimension)
    max_width = max(spec.shape[2] for spec in specs)
    
    # Pad all spectrograms to the same width
    padded_specs = []
    for spec in specs:
        # Calculate padding needed
        pad_amount = max_width - spec.shape[2]
        # Pad the time dimension (right side only)
        padded_spec = F.pad(spec, (0, pad_amount))
        padded_specs.append(padded_spec)
    
    # Stack into tensors
    specs_tensor = torch.stack(padded_specs)
    labels_tensor = torch.tensor(label_ids)
    
    return specs_tensor, labels_tensor, labels


def read_list(file_path):
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f.readlines())


def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=64):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_spectrograms,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_spectrograms,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_spectrograms,
    )
    
    return train_loader, val_loader, test_loader
