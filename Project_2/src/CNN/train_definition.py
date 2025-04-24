import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, 
                file_path='training_stats.csv', weight_decay=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Lists to store training and validation losses
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Calculate total number of batches
    total_batches = len(train_loader)
    print_interval = max(1, total_batches // 10)  # Print 10 times per epoch
    
    print(f"Training on {device} with {total_batches} batches per epoch")
    print(f"Will print progress every {print_interval} batches ({total_batches//print_interval} times per epoch)")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar for this epoch
        pbar = tqdm(enumerate(train_loader), total=total_batches, 
                     desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for batch_idx, batch in pbar:
            specs, labels, _ = batch
            specs, labels = specs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(specs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            batch_correct = (predicted == labels).sum().item()
            correct += batch_correct
            total += labels.size(0)
            
            # Update progress bar with current loss and accuracy
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'acc': f"{current_acc:.2f}%"
            })
            
            # Print detailed loss at specified intervals
            if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == total_batches:
                print(f"  Batch {batch_idx+1}/{total_batches} - Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%")
        
        # Calculate average loss for this epoch
        avg_train_loss = running_loss / total_batches
        train_losses.append(avg_train_loss)
        train_accuracies.append(correct / total * 100)

        # Validate the model
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        print("Validating...")
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=True)
            for batch in val_pbar:
                specs, labels, _ = batch
                specs, labels = specs.to(device), labels.to(device)
                
                outputs = model(specs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                batch_correct = (predicted == labels).sum().item()
                val_correct += batch_correct
                val_total += labels.size(0)
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'loss': f"{val_running_loss / (val_pbar.n + 1):.4f}",
                    'acc': f"{100. * val_correct / val_total:.2f}%"
                })
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = val_correct / val_total * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)  # Store as percentage
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print("-" * 80)
    
    # Save training statistics
    dataframe = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'train_loss': train_losses,
        'train_accuracy': [acc for acc in train_accuracies],  # Convert to percentage
        'val_loss': val_losses,
        'val_accuracy': [acc for acc in val_accuracies],  # Convert to percentage
    })
    dataframe.to_csv(file_path, index=False)
    
    # Plot training/validation curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    return train_losses, train_accuracies, val_losses, val_accuracies


def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training and validation loss/accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot([acc for acc in train_accuracies], label='Training Accuracy')
    ax2.plot([acc for acc in val_accuracies], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    # Set y-limits
    ax2.set_ylim(0, 100)
    ax2.axhline(y=5, color='r', linestyle='--', label='Random Threshold')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    # plt.show()


def evaluate_model(model, test_loader, train_loader):
    """(DOESN'T WORK YET) Evaluate the model on the test set"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    # Create confusion matrix
    train_dataset = train_loader.dataset
    num_classes = train_dataset.num_classes
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            specs, labels, _ = batch
            specs, labels = specs.to(device), labels.to(device)
            
            outputs = model(specs)
            _, predicted = outputs.max(1)
            
            # Update accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    # Calculate accuracy
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Add accuracy to training_stats.csv
    if os.path.exists('training_stats.csv'):
        stats_df = pd.read_csv('training_stats.csv')
        stats_df['test_accuracy'] = accuracy
        stats_df.to_csv('training_stats.csv', index=False)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    classes = train_dataset.labels
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    
    # Add values in cells
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, int(confusion_matrix[i, j]),
                    horizontalalignment="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    # Improve layout with tighter margins but ensure labels are visible
    plt.tight_layout()
    # Adjust subplot parameters to give specified padding
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    
    # Position labels closer to the axes
    plt.ylabel('True label', labelpad=10)
    plt.xlabel('Predicted label', labelpad=10)
    
    # Save figure with higher DPI and ensure all elements are within bounds
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=150)
    # plt.show()
    
    return accuracy, confusion_matrix


def main():
    pass

if __name__ == "__main__":
    main()