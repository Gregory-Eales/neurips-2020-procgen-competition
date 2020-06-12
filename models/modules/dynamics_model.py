import torch 
from torch import nn

from modules.rnn_submodule import RNNSubModule


class DynamicsModel(nn.Module):

    def __init__(self, action_size=15, latent_size=64, hidden_size=64+1):

        super(DynamicsModel, self).__init__()


        self.rnn = nn.GRUCell(latent_size+action_size, hidden_size)

    	

    def forward(self, latent_state, action):

    	out = torch.cat([latent_state, action])

    	out = self.rnn(out)

    	return out


if __name__ == "__main__":

	x = torch.randn(100, 79)
	rnn = nn.GRUCell(79, 65)

	print(rnn(x).shape)