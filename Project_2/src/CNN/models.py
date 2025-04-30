import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.utils.tensorboard import SummaryWriter

class AudioCNN(torch.nn.Module):
    def __init__(self, num_classes, n_mels=64):
        super(AudioCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Second convolutional block
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Third convolutional block
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fourth convolutional block
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate size after convolutions
        # After 4 max pooling layers with stride 2, dimensions are reduced by a factor of 2^4 = 16
        self.mel_reduced = n_mels // 16
        
        # We don't know the exact width after padding in the collate function,
        # so we'll use adaptive pooling to handle variable widths
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((self.mel_reduced, 8))
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(256 * self.mel_reduced * 8, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, x):
        # Input x shape: [batch, channels (1), mel_bands, time]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Apply adaptive pooling to get fixed output size
        x = self.adaptive_pool(x)
        
        # Flatten for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers with dropout
        x = self.dropout(torch.nn.functional.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


# Function to predict a single audio file
def predict_audio_file(file_path, model, device, dataset, n_mels=64, n_fft=1024, hop_length=512):
    '''
    NOT TESTED YET
    '''
    model.eval()
    
    # Load the audio file
    waveform, sample_rate = torchaudio.load(file_path)
    
    # Create transforms
    mel_transform = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    amp_to_db = T.AmplitudeToDB()
    
    # Convert to mel spectrogram
    mel_spec = mel_transform(waveform)
    mel_spec_db = amp_to_db(mel_spec)
    
    # Standardize using training set statistics if available
    if hasattr(dataset, 'spec_mean') and hasattr(dataset, 'spec_std'):
        mel_spec_db = (mel_spec_db - dataset.spec_mean) / dataset.spec_std
    
    # Add batch dimension
    mel_spec_db = mel_spec_db.unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        mel_spec_db = mel_spec_db.to(device)
        output = model(mel_spec_db)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = output.argmax(1).item()
        
    # Get the predicted class and probability
    predicted_class = dataset.labels[predicted_idx]
    probability = probabilities[0][predicted_idx].item()
    
    # Get top 3 predictions
    top3_values, top3_indices = torch.topk(probabilities, 3)
    top3_classes = [dataset.labels[idx] for idx in top3_indices[0].cpu().numpy()]
    top3_probs = top3_values[0].cpu().numpy()
    
    return {
        'predicted_class': predicted_class,
        'probability': probability,
        'top3_classes': top3_classes,
        'top3_probs': top3_probs,
        'waveform': waveform,
        'mel_spec_db': mel_spec_db[0],
        'sample_rate': sample_rate
    }
