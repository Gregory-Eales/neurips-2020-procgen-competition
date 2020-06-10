
from torch import nn
import torch

class SmallEncoder(nn.Module):

    def __init__(self):
        super(SmallEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=4, stride=2)

        self.l_mu = nn.Linear(256, 256)
        self.l_logvar = nn.Linear(256, 256)


    def forward(self, x):

        out = x

        out = self.conv1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = nn.functional.relu(out)
        out = self.conv3(out)
        out = nn.functional.relu(out)
        out = self.conv4(out)
        out = nn.functional.relu(out)

        out = out.reshape(-1, 256)

        logvar = self.l_logvar(out)
        mu = self.l_mu(out)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return mu + eps*std


class SmallDecoder(nn.Module):

    def __init__(self):
        super(SmallDecoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(256, 32, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(16, 8, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(8, 3, kernel_size=6, stride=2)

    def forward(self, x):


        out = x.reshape(-1, 256, 1, 1)

        out = self.deconv1(out)
        out = nn.functional.relu(out)
        out = self.deconv2(out)
        out = nn.functional.relu(out)
        out = self.deconv3(out)
        out = nn.functional.relu(out)
        out = self.deconv4(out)
        out = torch.sigmoid(out)

        return out

if __name__ == "__main__":


    x = torch.ones(1, 3, 64, 64)

    print(x.shape)

    encoder = SmallEncoder()
    decoder = SmallDecoder()
  

    y = encoder.forward(x)


    y = decoder.forward(y)

    print(y.shape)