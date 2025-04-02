import torch
import intel_extension_for_pytorch as ipex
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class ConvBaseLearner(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1):
        super(ConvBaseLearner, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(64 * 2 * 2, num_classes)

    def forward(self, x, params=None):
        if params is None:
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.dropout(x)
            x = self.pool(x)

            x = F.relu(self.bn2(self.conv2(x)))
            x = self.dropout(x)
            x = self.pool(x)

            x = F.relu(self.bn3(self.conv3(x)))
            x = self.dropout(x)
            x = self.pool(x)

            x = F.relu(self.bn4(self.conv4(x)))
            x = self.dropout(x)
            x = self.pool(x)

            x = x.view(x.size(0), -1)
            logits = self.fc(x)
        else:
            x = F.conv2d(x, params['conv1.weight'], params['conv1.bias'], padding=1)
            x = F.batch_norm(x, running_mean=None, running_var=None, weight=params['bn1.weight'],
                             bias=params['bn1.bias'], training=True)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=True)
            x = self.pool(x)

            x = F.conv2d(x, params['conv2.weight'], params['conv2.bias'], padding=1)
            x = F.batch_norm(x, running_mean=None, running_var=None, weight=params['bn2.weight'],
                             bias=params['bn2.bias'], training=True)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=True)
            x = self.pool(x)

            x = F.conv2d(x, params['conv3.weight'], params['conv3.bias'], padding=1)
            x = F.batch_norm(x, running_mean=None, running_var=None, weight=params['bn3.weight'],
                             bias=params['bn3.bias'], training=True)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=True)
            x = self.pool(x)

            x = F.conv2d(x, params['conv4.weight'], params['conv4.bias'], padding=1)
            x = F.batch_norm(x, running_mean=None, running_var=None, weight=params['bn4.weight'],
                             bias=params['bn4.bias'], training=True)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=True)
            x = self.pool(x)

            x = x.view(x.size(0), -1)
            logits = F.linear(x, params['fc.weight'], params['fc.bias'])
        return logits