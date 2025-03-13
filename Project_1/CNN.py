# Create CNN to classify CINIC-10 datase using PyTorch and tutorial from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#tensorboard --logdir=Project_1/runs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(0)

import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

def get_run_name(config):
    """Create a unique name for each experiment run"""
    timestamp = datetime.now().strftime("%d.%m-%H.%M")
    return f"{timestamp}---{config['layers']}-layers_lr{config['lr']}_batch{config['batch_size']}"


# Define the CNN
class CNN_ours(nn.Module):
    def __init__(self):
        super(CNN_ours, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)  # 3 input channels (RGB), 32 output channels, 3x3 kernel | 3x32x32 -> 32x30x30
        self.conv2 = nn.Conv2d(32, 32, 3)  # 32 input channels, 32 output channels, 3x3 kernel | 32x30x30 -> 32x28x28
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling | 32x28x28 -> 32x14x14
        self.conv3 = nn.Conv2d(32, 64, 3)  # 32 input channels, 64 output channels, 3x3 kernel | 32x14x14 -> 64x12x12
        self.conv4 = nn.Conv2d(64, 64, 3)  # 64 input channels, 64 output channels, 3x3 kernel | 64x12x12 -> 64x10x10

        self.fc1 = nn.Linear(64 * 5 * 5, 512) # 64x5x5 = 1600 -> 1600/2=800
        self.fc2 = nn.Linear(512, 256) 
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # Let's define the forward pass one layer at a time and add comments after each line to explain what is happening
        x = F.relu(self.conv1(x))  # Apply the first convolutional layer, then apply the ReLU activation function
        x = self.pool(F.relu(self.conv2(x)))  # Apply the second convolutional layer, then apply the ReLU activation function, then apply the pooling layer
        x = F.relu(self.conv3(x))  # Apply the third convolutional layer, then apply the ReLU activation function
        x = self.pool(F.relu(self.conv4(x)))  # Apply the fourth convolutional layer, then apply the ReLU activation function, then apply the pooling layer
        x = torch.flatten(x, 1)  # Flatten the output of the pooling layer
        x = F.relu(self.fc1(x))  # Apply the first fully connected layer
        x = F.relu(self.fc2(x))  # Apply the second fully connected layer
        x = self.fc3(x)  # Apply the third fully connected layer
        return x

if __name__ == "__main__":
    # Define hyperparameters as a dictionary for easy logging
    config = {
        'model': 'BASIC',
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 10,
        'early_stopping_patience': 5,
        'layers': 'basic',
        'subset_percentage': 0.01,  # Use 1% of data
        'use_subset': False  # Flag to enable/disable subsetting
    }

    # Define the device to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load the CINIC-10 dataset from data folder
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    batch_size = config['batch_size']

    # Print every 10% of the dataset
    if config['use_subset']:
        print_every = 90000*config['subset_percentage']//batch_size//10
        print_every = int(print_every)
    else:
        print_every = 90000//batch_size//10

    if print_every == 0:
        print_every = 1

    # Create a unique run name
    run_name = get_run_name(config)

    # Initialize TensorBoard writer
    if not config['model'] in os.listdir('Project_1/runs'):
        os.makedirs(f'Project_1/runs/{config["model"]}') 
    writer = SummaryWriter(f'Project_1/runs/{config["model"]}/{run_name}')

    PATH_to_folder = None

    print(f"Current path: {os.getcwd()}")
    # If the working directory is Project_1, then the path to the folder is 'Project_1/data/train''
    if 'Project_1' in os.listdir():
        PATH_to_folder = 'Project_1/data'
    # If the working directory is above Project_1, then the path to the folder is 'data/train'
    else:
        PATH_to_folder = 'data'

    trainset = torchvision.datasets.ImageFolder(root=os.path.join(PATH_to_folder, 'train'), transform=transform)

    valset = torchvision.datasets.ImageFolder(root=os.path.join(PATH_to_folder, 'valid'), transform=transform)

    # Create a subset of the data if needed
    if config['use_subset']:
        subset_percentage = config['subset_percentage']

        train_size = len(trainset)
        subset_size = int(train_size * subset_percentage)
        indices = np.random.choice(range(train_size), size=subset_size, replace=False)
        train_subset = Subset(trainset, indices)
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)

        # For validation data
        val_size = len(valset)
        subset_size_val = int(val_size * subset_percentage)
        indices_val = np.random.choice(range(val_size), size=subset_size_val, replace=False)
        val_subset = Subset(valset, indices_val)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

        print(f"Using {subset_size} training samples out of {train_size}")
        print(f"Using {subset_size_val} validation samples out of {val_size}")
    else:    
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Create the network
    net = CNN_ours().to(device)

    # Log model graph
    # example_input = torch.rand(1, 3, 32, 32).to(device)
    # writer.add_graph(net, example_input)

    critetion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(net.parameters(), lr=config['lr'])

    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs_stop = config['early_stopping_patience']


    print("Starting Training")
    # Train the network
    for epoch in tqdm(range(config['epochs'])):  # loop over the dataset multiple times
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
            loss = critetion(outputs, labels)
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

            # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', loss, epoch * len(trainloader) + i)       
            writer.add_scalar('Accuracy/train', train_accuracy, epoch * len(trainloader) + i)

            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every-1:  # print every x mini-batches
                avg_loss = running_loss / print_every
                print(f"\n[Epoch:{epoch + 1}, Batch {i + 1} / {len(trainloader)}] loss: {running_loss / print_every:.3f}, accuracy: {train_accuracy:.2f}%")
                running_loss = 0.0

        # Validate the network
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_accuracies = []

        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                images, labels = data[0].to(device), data[1].to(device)
                # forward
                outputs = net(images)
                loss = critetion(outputs, labels)
                val_loss += loss.item()
                
                # check predictions
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_accuracy = 100 * correct / total
                val_accuracies.append(val_accuracy)

                # Log metrics to TensorBoard
                writer.add_scalar('Loss/validation', loss, epoch * len(valloader) + i)
                writer.add_scalar('Accuracy/validation', val_accuracy , epoch * len(valloader) + i)

        val_loss /= len(valloader)
        val_accuracy = np.mean(val_accuracies)
        print(f"Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.2f}%")
   

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
    writer.add_hparams(
    {'lr': config['lr'], 'batch_size': config['batch_size'], 'epochs': config['epochs'], 'early_stopping_patience': config['early_stopping_patience'],
    'train_loss': np.mean(train_losses) ,'train_acc':np.mean(train_accuracies),'val_loss': val_loss, 'val_acc': val_accuracy},
    {'hparam/accuracy': val_accuracy, 'hparam/loss': val_loss}
    )
    writer.flush()
    writer.close()

    print('Finished Training: Validation Loss: {:.3f}, Validation Accuracy: {:.2f}%'.format(val_loss, val_accuracy))

    # Save the model
    if 'Project_1' in os.listdir():
        if not os.path.exists(f'Project_1/models/{config["model"]}'):
            os.makedirs(f'Project_1/models/{config["model"]}')
        PATH = f'Project_1/models/{run_name}.pth'
        torch.save(net.state_dict(), PATH)
    else:
        if not os.path.exists(f'models/{config["model"]}'):
            os.makedirs(f'models/{config["model"]}')
        PATH = f'models/{config["model"]}/{run_name}.pth'
        torch.save(net.state_dict(), PATH)
