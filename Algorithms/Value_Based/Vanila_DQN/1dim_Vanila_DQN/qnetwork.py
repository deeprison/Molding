import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 

class QNetwork(nn.Module):
    ''' Simple linear Q-Network. The architecture is, therefore, different from thg model in DQN paper.'''
    def __init__(self, 
                 input_feature: ("int: input state dimension"), 
                 action_dim: ("output: action dimensions"),
        ):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim

        self.linear1 = nn.Linear(input_feature, 256)
        self.linear2 = nn.Linear(256, 128) 
        self.linear3 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.linear1(x))
        x = self.linear3(self.relu(self.linear2(x)))
        return x 