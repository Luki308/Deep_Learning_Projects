import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from Project_1.src.from_scratch.CNN_definiton import CNN_ours
from Project_1.src.DataHelper import DataHelper

import torch
import torch.nn as nn

# create a function to run model from given path and evaluate it on test dataset

def evaluate_model(model_path, test_loader, device, save_path='Project_1/stats/model_performance.txt'):
    # Load the model
    model = CNN_ours()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Initialize variables to track total loss and correct predictions
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model on the test dataset
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)

            # Update total loss and correct predictions
            total_loss += loss.item() * data.size(0)  # Multiply by batch size to get total loss
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += data.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    # Save the model's performance metrics to a file
    with open(save_path, 'a') as f:
        f.write(f'Average Loss: {avg_loss:.4f}\n')
        f.write(f'Accuracy: {accuracy:.4f}\n')
    return avg_loss, accuracy

if __name__ == "__main__":
    # Example usage
    model_path = 'Project_1/models/BASIC_few_shots/default_BASIC_few_shots_epochs20_subset1pc_01.04-00.09_time-01.04-00.22.pth'  # Replace with your model path
    
    DataH = DataHelper(resize=32, batch_size=16, subset_fraction=0.01)
    trainloader, valloader, testloader = DataH.get_loaders()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_path = 'Project_1/stats/model_performance_1txt'
    avg_loss, accuracy = evaluate_model(model_path, testloader, device, save_path=save_path)
    print(f'Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    model_path = 'Project_1/models/BASIC_few_shots/default_BASIC_few_shots_epochs20_subset1pc_01.04-00.22_time-01.04-00.35.pth'
    avg_loss, accuracy = evaluate_model(model_path, testloader, device, save_path=save_path)
    print(f'Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    model_path = 'Project_1/models/BASIC_few_shots/default_BASIC_few_shots_epochs20_subset1pc_01.04-00.35_time-01.04-00.43.pth'
    avg_loss, accuracy = evaluate_model(model_path, testloader, device, save_path=save_path)
    print(f'Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')