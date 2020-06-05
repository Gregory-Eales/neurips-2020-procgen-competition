import torch

class Block(torch.nn.Module):

    def __init__(self, hparams):
        
        super(Block, self).__init__()

        self.hparams = hparams
    
        self.kernal_size = self.hparams.kernal_size
        self.num_channel = self.hparams.num_channels

        self.conv1 = torch.nn.Conv2d(self.num_channel, self.num_channel, kernel_size=self.kernal_size)
        self.conv2 = torch.nn.Conv2d(self.num_channel, self.num_channel, kernel_size=self.kernal_size)
        
        self.pad = torch.nn.ZeroPad2d(1)
        self.batch_norm = torch.nn.BatchNorm2d(self.num_channel)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):

        out = self.pad(x)
        out = self.conv1(out)
        out = self.batch_norm(out)
        out = self.relu(out)

        out = self.pad(x)
        out = self.conv2(out)
        out = self.batch_norm(out)
        out = out + x

        out = self.relu(out)

        return out