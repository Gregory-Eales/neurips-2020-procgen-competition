
from torch import nn
import torch

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=4, stride=2)


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

        return out


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2)

    def forward(self, x):

        out = x

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

    encoder = Encoder()
    decoder = Decoder()
  

    y = encoder.forward(x)

    print(y.shape)
    y = y.reshape(-1, 64*2*2, 1, 1)
    print(y.shape)

    y = decoder.forward(y)

    print(y.shape)