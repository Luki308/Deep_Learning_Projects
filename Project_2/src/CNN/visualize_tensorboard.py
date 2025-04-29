import torch
from models import AudioCNN
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import OrderedDict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_layer_parameters(model):
    # Dictionary to store layer names and their parameter counts
    layer_params = OrderedDict()
    
    # Iterate through named modules
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            layer_params[name] = params
    
    # Calculate total parameters and print table
    total_params = sum(layer_params.values())
    
    # Print header
    print("\nParameter Count by Layer:")
    print("-" * 50)
    print(f"{'Layer':<30}{'Parameters':<15}{'%Total':>10}")
    print("-" * 50)
    
    # Print each layer
    for name, params in layer_params.items():
        percentage = 100 * params / total_params
        print(f"{name:<30}{params:<15,}{percentage:>10.2f}%")
    
    # Print footer
    print("-" * 50)
    print(f"{'Total':<30}{total_params:<15,}{100:>10.2f}%")
    print("-" * 50)

def print_model_summary():
    model = AudioCNN(num_classes=10, n_mels=64)
    
    # Print model architecture
    print(model)
    
    # Print detailed parameter count by layer
    print_layer_parameters(model)
    
    # Print total parameter count
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    # Print output shape for each layer
    x = torch.randn(1, 1, 64, 100)  # [batch, channels, mel_bands, time]
    
    print("\nLayer Output Shapes:")
    print(f"Input: {x.shape}")
    
    x1 = model.conv1(x)
    print(f"Conv1: {x1.shape}")
    
    x2 = model.conv2(x1)
    print(f"Conv2: {x2.shape}")
    
    x3 = model.conv3(x2)
    print(f"Conv3: {x3.shape}")
    
    x4 = model.conv4(x3)
    print(f"Conv4: {x4.shape}")
    
    x5 = model.adaptive_pool(x4)
    print(f"Adaptive Pool: {x5.shape}")
    
    x5_flat = x5.view(x5.size(0), -1)
    print(f"Flattened: {x5_flat.shape}")
    
    x6 = torch.nn.functional.relu(model.fc1(x5_flat))
    print(f"FC1: {x6.shape}")
    
    x7 = model.fc2(x6)
    print(f"FC2 (Output): {x7.shape}")

def print_detailed_layer_info():
    model = AudioCNN(num_classes=10, n_mels=64)
    
    print("\nDetailed Layer Information:")
    print("=" * 80)
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)):
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            # Get input/output features for this layer
            if isinstance(module, torch.nn.Conv2d):
                layer_type = "Conv2d"
                details = f"in={module.in_channels}, out={module.out_channels}, " \
                          f"kernel={module.kernel_size}, stride={module.stride}, " \
                          f"padding={module.padding}"
            elif isinstance(module, torch.nn.Linear):
                layer_type = "Linear"
                details = f"in={module.in_features}, out={module.out_features}"
            elif isinstance(module, torch.nn.BatchNorm2d):
                layer_type = "BatchNorm2d"
                details = f"features={module.num_features}"
            
            print(f"{name} ({layer_type}):")
            print(f"  {details}")
            print(f"  Parameters: {params:,}")
            print("-" * 80)

if __name__ == "__main__":
    print_model_summary()
    print_detailed_layer_info()

# # Create model and dummy input
# model = AudioCNN(num_classes=10, n_mels=64)
# dummy_input = torch.randn(1, 1, 64, 100)  # [batch, channels, mel_bands, time]

# # Create SummaryWriter and add graph
# writer = SummaryWriter('runs/audio_cnn_architecture')
# writer.add_graph(model, dummy_input)
# writer.close()

# print("Run 'tensorboard --logdir=runs' and open http://localhost:6006 to view the model")