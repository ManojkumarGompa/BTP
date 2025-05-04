import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class Convolution(nn.Module):
    def __init__(self, input_shape):
        super(Convolution, self).__init__()
        
        # Input shape should be (C, H, W) - typically (4, 84, 84) for stacked frames
        self.input_shape = input_shape
        
        # Enhanced convolution layers
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Residual block
        self.res_block = ResidualBlock(128)
        
        # Calculate output dimensions
        self.conv_out_dims = self.calculate_conv_out_dims(input_shape)
        
        # Initialize weights orthogonally
        self._initialize_weights()
        
    def forward(self, x):
        # Ensure input has correct shape
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Apply residual block
        x = self.res_block(x)
        
        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)
        
        return x
        
    def calculate_conv_out_dims(self, input_shape):
        """Calculate output dimensions after convolution layers"""
        # Create dummy input to calculate output size
        dummy_input = torch.zeros(1, *input_shape)
        
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_block(x)
        
        return int(np.prod(x.shape[1:]))
        
    def convolute(self, state):
        """Legacy method for backward compatibility"""
        return self.forward(state)
        
    def _initialize_weights(self):
        """Initialize weights using orthogonal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)