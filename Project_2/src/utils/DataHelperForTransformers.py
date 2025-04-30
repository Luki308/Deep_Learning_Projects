import logging
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset


class ToLogMelSpec(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=64):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.db_transform = T.AmplitudeToDB()

    def forward(self, waveform):
        mel = self.mel_spec(waveform)  # [1, n_mels, time]
        # print("mel: ",mel.shape)
        # print("db_transform: ", db_transform.shape)
        mel_db = self.db_transform(mel)
        # print("mel_db: ", mel_db.shape)
        mel_db = mel_db.squeeze(0).transpose(0, 1)  # [time, n_mels] (aka [T, F])
        # print("mel_db squeeze: ", mel_db.shape)
        return mel_db


def collate_fn_spec(batch, target_time_steps=500):
    specs, labels = zip(*batch)
    padded_specs = []

    for spec in specs:
        t, f = spec.shape
        if t < target_time_steps:
            pad_amt = target_time_steps - t
            padded = F.pad(spec, (0, 0, 0, pad_amt))  # pad along time (rows)
        else:
            padded = spec[:target_time_steps, :]
        padded_specs.append(padded)

    specs_tensor = torch.stack(padded_specs)  # [B, T, F]
    return specs_tensor, labels


def read_list(file_path):
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f.readlines())


target_classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence"]


def assign_label(path):
    label = path.parent.name.lower()
    if label in target_classes:
        return label
    elif label == "_background_noise_":
        return "nope"
    else:
        return "unknown"


class SpeechCommandsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        waveform, _ = torchaudio.load(str(path))
        label = assign_label(path)

        if self.transform:
            spec = self.transform(waveform)
        else:
            raise ValueError("Transform is required for this model.")

        return spec, label

    def targets_true(self):
        return [path.parent.name for path in self.file_list]

    def targets_used(self):
        return [assign_label(path) for path in self.file_list]


def stratified_subset(dataset, subset_fraction):
    """Ensures class balance in the subset."""
    targets = dataset.targets_used()
    train_indices, _ = train_test_split(range(len(targets)), train_size=subset_fraction, stratify=targets,
                                        random_state=3371)
    return Subset(dataset, train_indices)


def get_data_loaders_2(data_dir='../../data/train', batch_size=32, shuffle_train=True, target_classes=None,
                       subset_fraction=None):
    data_dir = Path(data_dir).resolve()

    val_list = read_list(data_dir / 'validation_list.txt')
    test_list = read_list(data_dir / 'testing_list.txt')

    dataset_path = data_dir / 'audio'
    all_data = {'training': [], 'validation': [], 'testing': []}
    unknown_data = {'training': [], 'validation': [], 'testing': []}

    for audio_path in dataset_path.rglob("*.wav"):
        label = assign_label(audio_path)

        # Filter by target_classes if specified
        if target_classes and label not in target_classes:
            continue

        relative_path = audio_path.relative_to(dataset_path).as_posix()

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



    # print counts for debugging
    logging.info(f"Found {len(all_data['training'])} training files from classes: {target_classes or 'all'}")
    logging.info(f"Found {len(all_data['validation'])} validation files")
    logging.info(f"Found {len(all_data['testing'])} testing files")

    mel_transform = ToLogMelSpec()

    train_dataset = SpeechCommandsDataset(all_data['training'], transform=mel_transform)
    val_dataset = SpeechCommandsDataset(all_data['validation'], transform=mel_transform)
    test_dataset = SpeechCommandsDataset(all_data['testing'], transform=mel_transform)
    logging.info(f"classes count: ")
    targets_used = train_dataset.targets_used()
    logging.info(Counter(targets_used).items())

    if subset_fraction:
        train_dataset = stratified_subset(train_dataset, subset_fraction)
        val_dataset = stratified_subset(val_dataset, subset_fraction)
        test_dataset = stratified_subset(test_dataset, subset_fraction)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, collate_fn=collate_fn_spec)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn_spec)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn_spec)

    return train_loader, val_loader, test_loader


def balance_classes_and_join(all_data, target_classes, unknown_data):
    for name in ['training', 'validation', 'testing']:
        targets = [path.parent.name.lower() for path in unknown_data[name]]
        fraction = len(all_data[name]) / (len(unknown_data[name]) * (len(target_classes) - 1))
        unknown_data[name], _ = train_test_split(unknown_data[name], train_size=fraction, stratify=targets,
                                                     random_state=3371)
        all_data[name].extend(unknown_data[name])
    return all_data


def build_label_mapping(loader):
    # Go over the training data to build a set of labels
    label_set = set()
    for _, labels in loader:
        label_set.update(labels)
    label_to_idx = {label: idx for idx, label in enumerate(sorted(label_set))}
    return label_to_idx


def encode_labels(labels, label_to_idx):
    return torch.tensor([label_to_idx[label] for label in labels])


# example
if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_data_loaders_2(subset_fraction=0.1)
    i = 0
    print(f"Number of training files: {len(train_loader)}")

    for batch in train_loader:
        specs, labels = batch
        print(f"Waveform shape: {specs.shape}, Labels: {labels}")
        print(specs[0].mean(), specs[0].std(), specs[0].shape)
        i += 1
        if i == 1:
            break  # just show one batch for demonstration
