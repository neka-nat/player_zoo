import torch
import torch.nn as nn
import torch.nn.functional as F
from libs import utils

class DQN(nn.Module):
    def __init__(self, n_action, input_shape=(4, 84, 84)):
        super(DQN, self).__init__()
        self.n_action = n_action
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        r, c = utils.outsize(input_shape[1:], 8, 0, 4)
        r, c = utils.outsize((r, c), 4, 0, 2)
        r, c = utils.outsize((r, c), 3)
        self.fc1 = nn.Linear(r * c * 64, 512)
        self.fc2 = nn.Linear(512, self.n_action)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)
