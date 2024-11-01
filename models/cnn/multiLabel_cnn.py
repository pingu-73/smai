import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLabelCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5, num_conv_layers=2):
        super(MultiLabelCNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        
        # initial Conv Layer
        self.conv_layers.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1))
        
        # additional Conv Layers
        for i in range(1, num_conv_layers):
            in_channels = 32 if i == 1 else 64
            self.conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1))
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_rate)

        dummy_input = torch.randn(1, 1, 28, 28)
        conv_out_size = self._get_conv_output(dummy_input)

        self.fc1 = nn.Linear(conv_out_size, 128)
        self.fc2 = nn.Linear(128, num_classes)  # o/p layer for multi-label classification

        # loss function
        self.loss_function = nn.BCEWithLogitsLoss()

    def _get_conv_output(self, x):
        for conv in self.conv_layers:
            x = self.pool(F.relu(conv(x)))
        return x.numel()

    def forward(self, x):
        feature_maps = []  # intermediate feature maps
        
        # pass through each convolutional layer
        for conv in self.conv_layers:
            x = F.relu(conv(x))
            feature_maps.append(x) 
            x = self.pool(x) 
            
        # Flatten the output
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        logits = self.fc2(x) #no activation
        return logits, feature_maps

    def calculate_loss(self, outputs, labels): #binary cross entropy loss
        loss = self.loss_function(outputs, labels.float())
        return loss

    def calculate_accuracy(self, outputs, labels, threshold=0.5):
        probabilities = torch.sigmoid(outputs) #sigmod for probabilities
        predicted = (probabilities > threshold).float()  # binarize predictions

        correct = (predicted == labels).sum().item()  # count correct predictions
        total = labels.numel()  # total labels
        accuracy = correct / total
        return accuracy
