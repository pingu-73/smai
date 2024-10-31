import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNModel(nn.Module):
    def __init__(self, num_classes=10, task_type='classification', dropout_rate=0.5, num_conv_layers=2):
        super(CNNModel, self).__init__()
        self.task_type = task_type
        self.conv_layers = nn.ModuleList()
        
        # initial Conv Layer
        self.conv_layers.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1))
        
        # additional conv layers (based on num_conv_layers)
        for i in range(1, num_conv_layers):
            in_channels = 32 if i == 1 else 64
            self.conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1))
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_rate)

        dummy_input = torch.randn(1, 1, 28, 28)
        conv_out_size = self._get_conv_output(dummy_input)

        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, num_classes if task_type == 'classification' else 1)

        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()

    def _get_conv_output(self, x):
        for conv in self.conv_layers:
            x = self.pool(F.relu(conv(x)))
        return x.numel()

    def forward(self, x):
        feature_maps = []  # store intermediate feature maps
        
        # pass through each convolutional layer
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            feature_maps.append(x) 
            x = self.pool(x) 
            
        # flatten the output
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        

        if self.task_type == 'classification':
            logits = self.fc2(x)
            return logits, feature_maps
        else:
            regression_output = self.fc2(x)
            return regression_output, feature_maps

    def calculate_loss(self, outputs, labels):
        if self.task_type == 'classification':
            loss = self.classification_loss(outputs, labels)
        else:
            loss = self.regression_loss(outputs.squeeze(), labels.float())
        return loss

    def calculate_accuracy(self, outputs, labels):
        if self.task_type == 'classification':
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total
            return accuracy
        else:
            return None