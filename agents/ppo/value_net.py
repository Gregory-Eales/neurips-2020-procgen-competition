import torch
import numpy as np
from tqdm import tqdm

class ValueNetwork(torch.nn.Module):

    def __init__(self, hparams):

        super(ValueNetwork, self).__init__()
        self.hparams = hparams

        self.input_dims = self.hparams.input_dims
        self.output_dims = self.hparams.output_dims

        self.initialize_network()
        self.loss = torch.nn.MSELoss()

    # initialize network
    def initialize_network(self):

        # define convolutional network components
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=7)

		# define linear network components
        self.fc1 = torch.nn.Linear(self.input_dims, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, self.output_dims)
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        out = torch.Tensor(x).to(self.device)

        out = self.conv1(out)
        out = self.tanh(out)
        out = self.conv2(out)
        out = self.tanh(out)
        out = self.conv3(out)
        out = self.tanh(out)
        out = self.conv4(out)
        out = self.tanh(out)

        out = out.reshape(-1, 64)

        out = self.fc1(out)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.relu(out)
        
        return out.to(torch.device('cpu:0'))
