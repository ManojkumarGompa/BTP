import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from convol.convolution import Convolution

class Critic(nn.Module):
    def __init__(self, state_dim, action_size, hidden1_dim=512, hidden2_dim=256):
        super(Critic, self).__init__()
        
        # Convolution for feature extraction
        self.convolution = Convolution(state_dim)
        conv_out_dims = self.convolution.calculate_conv_out_dims(state_dim)
        
        # Action embedding
        self.action_embedding = nn.Linear(action_size, 64)
        
        # Combined layers - increased from the small hidden1_dim=6
        self.fc1 = nn.Linear(conv_out_dims + 64, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state, action):
        """
        Forward pass through the critic network
        state: batch of states (B, C, H, W)
        action: action probabilities (B, action_size)
        """
        # Process state through convolution
        if state.dim() == 4 and state.shape[1] == 210:
            state = state.permute(0, 3, 1, 2)
        
        conv_state = self.convolution(state)
        
        # Process action (assuming action is action probabilities or one-hot)
        action_emb = self.relu(self.action_embedding(action))
        
        # Concatenate state features and action embedding
        combined = torch.cat([conv_state, action_emb], dim=1)
        
        # Pass through fully connected layers
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        
        return value