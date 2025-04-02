import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from few_shots.Base import ConvBaseLearner


class MetaOptimizer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=20, num_layers=2):
        super(MetaOptimizer, self).__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * num_layers + [1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, grad):
        grad = grad.unsqueeze(-1)
        update = self.net(grad)
        return update.squeeze(-1)
