from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F


class FC_Net(nn.Module):
    def __init__(self):
        super(FC_Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 300)
        self.fc5 = nn.Linear(300, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        x = self.fc5(x)

        return x

