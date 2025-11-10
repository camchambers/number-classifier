"""
Digit Classifier Model Architecture
"""
import torch
import torch.nn as nn


class DigitClassifier(nn.Module):
    """
    Convolutional Neural Network for digit classification (0-9)
    
    Architecture:
    - 3 Convolutional layers (32, 64, 128 filters)
    - MaxPooling after each conv layer
    - Dropout for regularization
    - 3 Fully connected layers
    """
    
    def __init__(self):
        super(DigitClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 3 pooling layers: 28 -> 14 -> 7 -> 3 (floor division)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Conv layers with pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
