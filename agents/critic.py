import torch.nn as nn
import torch
from convol.convolution import Convolution
class Critic(nn.Module):
    def __init__(self, state_dim,action_dim, output_dim=1, hidden1_dim=6, hidden2_dim=6):
        super(Critic, self).__init__()

        self.q_convolution = Convolution(state_dim)
        conv_out_dims = self.q_convolution.calculate_conv_out_dims(state_dim)

        input_dim = conv_out_dims + action_dim
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        # self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden1_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, state , action):
        conv_state = self.q_convolution.convolute(state)
        # if action.shape[0]!=256:
        #     action = torch.Tensor.flatten(action)
        #     x  = torch.cat([conv_state, action], dim=0)
        # else:
        x  = torch.cat([conv_state, action], dim=1)

        # print("Action: ",action.shape,flush=True)
        # print(conv_state)
        # print(action)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
