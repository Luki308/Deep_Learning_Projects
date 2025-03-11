# Create CNN to classify CIFAR-10 datase using PyTorch and tutorial from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)

# Define the CNN
class CNN_ours(nn.Module):
    def __init__(self):
        super(CNN_ours, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3) # 3 input channels, 6 output channels, 3x3 kernel | 3x32x32 -> 6x30x30
        self.pool = nn.MaxPool2d(2, 2) # 2x2 pooling | 6x30x30 -> 6x15x15
        self.conv2 = nn.Conv2d(6, 16, 3) # 6 input channels, 16 output channels, 3x3 kernel | 6x15x15 -> 16x13x13

        self.fc1 = nn.Linear(16 * 6 * 6, 288) # 16x6x6 = 576 -> 576/2=288
        self.fc2 = nn.Linear(288, 288) 
        self.fc3 = nn.Linear(288, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Let's define the forward pass one layer at a time and add comments after each line to explain what is happening
        x = self.pool(torch.relu(self.conv1(x))) # Apply the first convolutional layer, then apply the ReLU activation function, then apply the pooling layer
        x = self.pool(torch.relu(self.conv2(x))) # Analogous to the previous line, but with the second convolutional layer
        x = torch.flatten(x, 1) # Flatten the output of the second convolutional layer
        x = F.relu(self.fc1(x)) # Apply the first fully connected layer
        x = F.relu(self.fc2(x)) # Apply the second fully connected layer
        x = self.fc3(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    # Define the device to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    # Load the CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='Project_1/data', train=True, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


    testset = torchvision.datasets.CIFAR10(root='Project_1/data', train=False, download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Create the network
    net = CNN_ours().to(device)

    critetion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(net.parameters(), lr=0.001)

    # Train the network
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = critetion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")

    PATH = 'Project_1/models/cifar_net.pth'
    torch.save(net.state_dict(), PATH)
