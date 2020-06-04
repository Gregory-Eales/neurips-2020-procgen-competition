import torch
from tqdm import tqdm
import numpy as np

class PolicyNetwork(torch.nn.Module):

    def __init__(self, hparams):

        super(PolicyNetwork, self).__init__()
        self.hparams = hparams

        self.in_dim = self.hparams.in_dim
        self.out_dim = self.hparams.out_dim
        self.epsilon = self.hparams.epsilon

        self.define_network()
        self.prev_params = self.parameters()

    def define_network(self):

        self.relu = torch.nn.ReLU()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=2)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=7)

        self.l1 = torch.nn.Linear(self.in_dim, 64)
        self.l2 = torch.nn.Linear(64, 64)
        self.l3 = torch.nn.Linear(64, self.out_dim)

    def loss(self, r_theta, advantages):

        clipped_r = torch.clamp(r_theta, 1.0 - self.epsilon, 1.0 + self.epsilon)
        return torch.min(r_theta*advantages, clipped_r*advantages)

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
       

        out = self.l1(out)
        out = self.tanh(out)
        out = self.l2(out)
        out = self.tanh(out)
        out = self.l3(out)
        out = self.sigmoid(out)

        return out.to(torch.device('cpu:0'))
