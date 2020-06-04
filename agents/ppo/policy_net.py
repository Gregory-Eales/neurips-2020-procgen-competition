import torch
from tqdm import tqdm
import numpy as np

class PolicyNetwork(torch.nn.Module):

    def __init__(self, alpha, in_dim, out_dim, epsilon=0.1):

        super(PolicyNetwork, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.epsilon = epsilon
        self.define_network()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
        self.to(self.device)
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


def main():

    t1 = torch.ones(1, 3)
    pn = PolicyNetwork(0.01, 3, 1)
    print(pn(t1))
    print(pn.parameters())


if __name__ == "__main__":
    main()