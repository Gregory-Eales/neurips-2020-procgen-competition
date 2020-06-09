import torch
from torch import nn


class ValueHead(nn.Module):

    def __init__(self, num_hidden):

        super(ValueHead, self).__init__()

        self.fc1 = nn.Linear(num_hidden, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 1)

    def forward(self, x):

        out = x

        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)

        return out