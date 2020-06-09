import torch
from torch import nn


class PolicyHead(nn.Module):

    def __init__(self, num_hidden, num_outputs):

        super(PolicyHead, self).__init__()

        self.fc1 = nn.Linear(num_hidden, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):

        out = x

        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        

        return out