import torch
from torch import nn


class LinearResBlock(nn.Module):

    def __init__(self, num_hidden):

        super(LinearResBlock, self).__init__()


        self.fc1 = nn.Linear(num_hidden, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)


    def forward(self, x):

        out = x

        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out = nn.functional.relu(out)

        return out + x