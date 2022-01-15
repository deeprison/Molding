import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_action):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.v1 = nn.Linear(1600,512)
        self.v2 = nn.Linear(512,1)
        self.p1 = nn.Linear(1600,512)
        self.p2 = nn.Linear(512,n_action)
        
        self.act_map = nn.ReLU()
    
    def forward(self, x):
        x = self.act_map(self.conv1(x/255))
        x = self.act_map(self.conv2(x))
        x = self.act_map(self.conv3(x))
        
        batch, _, _, _ = x.size()

        v = self.act_map(self.v1(x.view(batch, -1)))
        v = self.v2(v)

        p = self.act_map(self.p1(x.view(batch, -1)))
        p = self.p2(p)

        return p, v