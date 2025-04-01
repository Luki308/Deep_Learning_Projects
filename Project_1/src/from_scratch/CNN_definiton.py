import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Define the CNN
class CNN_ours(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(CNN_ours, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)  # 3 input channels (RGB), 32 output channels, 3x3 kernel | 3x32x32 -> 32x30x30
        self.conv2 = nn.Conv2d(32, 32, 3)  # 32 input channels, 32 output channels, 3x3 kernel | 32x30x30 -> 32x28x28
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling | 32x28x28 -> 32x14x14
        self.conv3 = nn.Conv2d(32, 64, 3)  # 32 input channels, 64 output channels, 3x3 kernel | 32x14x14 -> 64x12x12
        self.conv4 = nn.Conv2d(64, 64, 3)  # 64 input channels, 64 output channels, 3x3 kernel | 64x12x12 -> 64x10x10

        self.fc1 = nn.Linear(64 * 5 * 5, 512) # 64x5x5 = 1600 -> 1600/2=800
        self.fc2 = nn.Linear(512, 256) 
        self.fc3 = nn.Linear(256, 10)
        
        # Add dropout layers for regularization
        self.dropout1 = nn.Dropout2d(dropout_rate)  # Spatial dropout for convolutional layers
        self.dropout2 = nn.Dropout(dropout_rate)    # Regular dropout for fully connected layers

    def forward(self, x):
        # Let's define the forward pass one layer at a time and add comments after each line to explain what is happening
        x = F.relu(self.conv1(x))  # Apply the first convolutional layer, then apply the ReLU activation function
        x = self.pool(F.relu(self.conv2(x)))  # Apply the second convolutional layer, then apply the ReLU activation function, then apply the pooling layer
        x = self.dropout1(x)  # Apply spatial dropout after first pooling layer
        
        x = F.relu(self.conv3(x))  # Apply the third convolutional layer, then apply the ReLU activation function
        x = self.pool(F.relu(self.conv4(x)))  # Apply the fourth convolutional layer, then apply the ReLU activation function, then apply the pooling layer
        x = self.dropout1(x)  # Apply spatial dropout after second pooling layer
        
        x = torch.flatten(x, 1)  # Flatten the output of the pooling layer
        x = F.relu(self.fc1(x))  # Apply the first fully connected layer
        x = self.dropout2(x)  # Apply dropout after first fully connected layer
        
        x = F.relu(self.fc2(x))  # Apply the second fully connected layer
        x = self.dropout2(x)  # Apply dropout after second fully connected layer
        
        x = self.fc3(x)  # Apply the third fully connected layer
        return x

if __name__ == "__main__":
    # Test the CNN_ours class
    model = CNN_ours(dropout_rate=0.2)

    # Print the model architecture
    print(model)

    # Print the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")
    
    summary(model, (3, 32, 32), device="cpu")  # Print the model summary for an input size of (3, 32, 32)