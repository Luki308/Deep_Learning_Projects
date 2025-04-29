import torch
import torchaudio
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from transformers import Wav2Vec2FeatureExtractor
from sklearn.model_selection import train_test_split

class SpeechCommandsWav2VecDataset(Dataset):
    def __init__(
        self, 
        file_list: List[Path], 
        feature_extractor: Optional[Wav2Vec2FeatureExtractor] = None,
        max_length: int = 16000,  # 1 second at 16kHz
        sample_rate: int = 16000,
        return_attention_mask: bool = True
    ):
        """
        Dataset for Wav2Vec fine-tuning on Speech Commands
        
        Args:
            file_list: List of audio file paths
            feature_extractor: Wav2Vec feature extractor (will be loaded if None)
            max_length: Maximum length of audio in samples
            sample_rate: Target sample rate
            return_attention_mask: Whether to return attention masks
        """
        self.file_list = file_list
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.return_attention_mask = return_attention_mask
        
        # Load feature extractor if not provided
        if feature_extractor is None:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        else:
            self.feature_extractor = feature_extractor
        
        # Get unique label names and create a mapping to indices
        # self.labels = sorted(list(set(path.parent.name for path in file_list)))
        # if '_background_noise_' in self.labels:
        #     self.labels.remove('_background_noise_')
        # self.label_to_index = {label: i for i, label in enumerate(self.labels)}
        # self.index_to_label = {i: label for label, i in self.label_to_index.items()}

        self.target_classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        self.labels = self.target_classes + ["unknown", "silence"]
        # Map original folder names to our target labels
        self.label_to_index = {label: i for i, label in enumerate(self.labels)}
        self.index_to_label = {i: label for label, i in self.label_to_index.items()}
        
        print(f"Found {len(self.labels)} unique classes: {self.labels}")
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        path = self.file_list[idx]
        
        try:
            # Load audio using torchaudio
            waveform, sample_rate = torchaudio.load(path, normalize=True)
            
            # Convert to mono if stereo
            if waveform.dim() > 1:
                waveform = waveform.mean(dim=0)
            
            # Resample if needed 
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Process with feature extractor (convert to numpy first)
            waveform_np = waveform.numpy()
            inputs = self.feature_extractor(
                waveform_np,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            # Extract input values and squeeze first dimension
            input_values = inputs.input_values.squeeze(0)
            
            # Get the label
            original_label = path.parent.name

            if original_label in self.target_classes:
                label = original_label
            else:
                label = "unknown"            

            label_idx = self.label_to_index.get(label, self.label_to_index["unknown"])
            
            # Create result dictionary
            result = {
                "input_values": input_values,
                "label": label_idx,
                "label_name": label,
            }
            
            # Add attention mask if needed
            if self.return_attention_mask and "attention_mask" in inputs:
                result["attention_mask"] = inputs.attention_mask.squeeze(0)
            
            return result
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a placeholder instead of failing
            dummy_input = torch.zeros(self.max_length)
            return {
                "input_values": dummy_input,
                "label": 0,  # Use first class as fallback
                "label_name": self.labels[0],
                "attention_mask": torch.ones(self.max_length) if self.return_attention_mask else None,
                "error": str(e)
            }
    
    @property
    def num_classes(self):
        return len(self.labels)


class Wav2VecDataCollator:
    """
    Data collator for Wav2Vec that handles dynamic padding and errors
    """
    def __init__(self, feature_extractor=None, padding=True, return_tensors="pt", return_attention_mask=True):
        self.feature_extractor = feature_extractor
        self.padding = padding
        self.return_tensors = return_tensors
        self.return_attention_mask = return_attention_mask
    
    def __call__(self, features):
        try:
            # Extract values
            input_values = [feature["input_values"] for feature in features]
            labels = [feature["label"] for feature in features]
            
            # Check if we have any tensors with NaNs or Infs
            for i, tensor in enumerate(input_values):
                if isinstance(tensor, torch.Tensor):
                    has_nan = torch.isnan(tensor).any()
                    has_inf = torch.isinf(tensor).any()
                    if has_nan or has_inf:
                        # Replace with zeros if there are problems
                        input_values[i] = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Handle padding manually if feature_extractor is not available
            if self.feature_extractor is None:
                # Get the max length
                max_length = max(len(x) for x in input_values)
                
                # Create padded input values
                padded_inputs = []
                attention_mask = []
                
                for values in input_values:
                    length = len(values)
                    padding_length = max_length - length
                    
                    if padding_length > 0:
                        padded_values = torch.nn.functional.pad(values, (0, padding_length))
                        mask = torch.cat([torch.ones(length), torch.zeros(padding_length)])
                    else:
                        padded_values = values
                        mask = torch.ones(length)
                    
                    padded_inputs.append(padded_values)
                    attention_mask.append(mask)
                
                batch = {
                    "input_values": torch.stack(padded_inputs),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
                
                if self.return_attention_mask:
                    batch["attention_mask"] = torch.stack(attention_mask)
            else:
                # Use the feature extractor for padding
                batch = self.feature_extractor.pad(
                    {"input_values": input_values},
                    padding=self.padding,
                    return_tensors=self.return_tensors,
                    return_attention_mask=self.return_attention_mask,
                )
                batch["labels"] = torch.tensor(labels, dtype=torch.long)
            
            # Ensure all tensors are on CPU and contiguous
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    batch[key] = tensor.detach().contiguous()
            
            return batch
            
        except Exception as e:
            print(f"Error in collation: {e}")
            # Create a minimal valid batch to avoid breaking the training loop
            batch_size = len(features)
            fallback_batch = {
                "input_values": torch.zeros((batch_size, 16000)),
                "labels": torch.zeros(batch_size, dtype=torch.long),
            }
            if self.return_attention_mask:
                fallback_batch["attention_mask"] = torch.ones((batch_size, 16000))
            return fallback_batch


def read_file_list(file_path):
    """Read a file list into a set of strings"""
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f.readlines())
    
def assign_label(path, target_classes):
    label = path.parent.name.lower()
    if label in target_classes:
        return label
    elif label == "_background_noise_":
        return "nope"
    else:
        return "unknown"

def balance_classes_and_join(all_data, target_classes, unknown_data):
    for name in ['training', 'validation', 'testing']:
        targets = [path.parent.name.lower() for path in unknown_data[name]]
        fraction = len(all_data[name]) / (len(unknown_data[name]) * (len(target_classes) - 1))
        unknown_data[name], _ = train_test_split(unknown_data[name], train_size=fraction, stratify=targets,
                                                     random_state=3371)
        all_data[name].extend(unknown_data[name])
    return all_data

def split_dataset_by_lists(
    dataset_path: Path,
):
    """
    Split dataset using the official validation and testing lists
    
    Args:
        dataset_path: Path to the dataset root
        validation_list_path: Path to validation_list.txt
        testing_list_path: Path to testing_list.txt
        selected_commands: Optional list of command categories to include
        fallback_to_random: Whether to fall back to random splitting if lists are not found
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    # Ensure correct paths
    audio_folder_path = dataset_path / 'audio'

    if not audio_folder_path.exists():
        audio_folder_path = dataset_path

    val_list = read_file_list(dataset_path / 'validation_list.txt')
    test_list = read_file_list(dataset_path / 'testing_list.txt')

    print(f"Validation files: {len(val_list)}")
    print(f"Testing files: {len(test_list)}")

    target_classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence"]

    # Find all audio files
    all_data = {'training': [], 'validation': [], 'testing': []}
    unknown_data = {'training': [], 'validation': [], 'testing': []}
    print(f"Looking for audio files in: {audio_folder_path}")
    
    for audio_path in audio_folder_path.rglob("*.wav"):
        label = assign_label(audio_path, target_classes)
        relative_path = audio_path.relative_to(audio_folder_path).as_posix()
        # if label != "unknown":
        # if audio_path.parent.name in target_classes:
        #     print(f"Processing {relative_path} with label {label}")
        #     print(audio_path.parent.name)

        print(f"Processing {relative_path} with label {label}")
        print(audio_path.parent.name)
        # Filter by target_classes if specified
        if target_classes and label not in target_classes:
            continue


        if label == "unknown":
            if relative_path in val_list:
                unknown_data['validation'].append(audio_path)
            elif relative_path in test_list:
                unknown_data['testing'].append(audio_path)
            else:
                unknown_data['training'].append(audio_path)
            continue

        if relative_path in val_list:
            all_data['validation'].append(audio_path)
        elif relative_path in test_list:
            all_data['testing'].append(audio_path)
        else:
            all_data['training'].append(audio_path)

        if 'unknown' in target_classes:
            all_data = balance_classes_and_join(all_data, target_classes, unknown_data) 

        return all_data['training'], all_data['validation'], all_data['testing']



def create_wav2vec_datasets(
    dataset_path: Path,
    validation_list_path: str,
    testing_list_path: str,
    selected_commands: Optional[List[str]] = None,
    feature_extractor: Optional[Wav2Vec2FeatureExtractor] = None,
    max_length: int = 16000,
    sample_rate: int = 16000
):
    """
    Create train, validation and test datasets for Wav2Vec fine-tuning
    
    Args:
        dataset_path: Path to the dataset root
        validation_list_path: Path to validation_list.txt
        testing_list_path: Path to testing_list.txt
        selected_commands: Optional list of command categories to include
        feature_extractor: Wav2Vec feature extractor (will be loaded if None)
        max_length: Maximum length of audio in samples
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Split the dataset
    train_files, val_files, test_files = split_dataset_by_lists(dataset_path)
    
    # Create datasets
    train_dataset = SpeechCommandsWav2VecDataset(
        train_files, feature_extractor, max_length, sample_rate
    )
    
    # Reuse the same feature extractor for all datasets
    if feature_extractor is None and hasattr(train_dataset, 'feature_extractor'):
        feature_extractor = train_dataset.feature_extractor
    
    val_dataset = SpeechCommandsWav2VecDataset(
        val_files, feature_extractor, max_length, sample_rate
    )
    
    test_dataset = SpeechCommandsWav2VecDataset(
        test_files, feature_extractor, max_length, sample_rate
    )
    
    return train_dataset, val_dataset, test_dataset
