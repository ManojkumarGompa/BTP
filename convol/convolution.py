import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class Convolution(nn.Module):
    def __init__(self, input_dim):
        """
        input_dim: tuple of (H, W, C)
        Internally converted to (C, H, W) for PyTorch Conv2d
        """
        super(Convolution, self).__init__()

        # print("Input Dim:", input_dim, flush=True)
        input_dim = self._to_chw(input_dim)
        # print("Adjusted Input Dim (C, H, W):", input_dim, flush=True)

        self.conv1 = nn.Conv2d(in_channels=input_dim[0], out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=4, stride=1)

        self.conv_out_dims = self.calculate_conv_out_dims(input_dim)
        print("Conv Out Dims:", self.conv_out_dims, flush=True)

    def _to_chw(self, dims):
        """Converts (H, W, C) → (C, H, W)"""
        return (dims[2], dims[0], dims[1])

    def calculate_conv_out_dims(self, input_dims):
        """
        Calculates the flattened output dimension after conv layers.
        input_dims must be (C, H, W)
        """
        input_dims = self._to_chw(input_dims) if input_dims[0] in (210, 160) else input_dims

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dims)  # (1, C, H, W)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            return int(np.prod(x.size()))

    def convolute(self, state):
        """
        Applies conv layers and ReLU, then flattens.
        Expects input state as (B, H, W, C) → converts to (B, C, H, W)
        """
        # print("State Shape:", state.shape, flush=True)
        
        # Ensure the input state is in (B, C, H, W) format
        if state.dim() == 4 and state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)  # NHWC → NCHW
        
        # If the input is a single image without batch size, add a batch dimension
        if state.dim() == 3:
            state = state.unsqueeze(0)
            state=state.permute(0,3,1,2)  # Add batch dimension: (1, C, H, W)
        # print the state shape here
        # print("State Shape after permute:", state.shape, flush=True)
        # Now, apply the conv layers and ReLU
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output
        return x.reshape(x.size(0), -1)  # Flatten the output to (batch_size, -1)

 