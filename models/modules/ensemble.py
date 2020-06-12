import torch 
from torch import nn

class Ensemble(nn.Module):

    def __init__(self, num_ensembles, obs_space, action_space):

        super(Ensemble, self).__init__()


        self.num_ensembles = num_ensembles
        self.ensembles = nn.ModuleList()
        self.disagreement = None


        for i in range(self.num_ensembles):
            self.ensembles.append(nn.Linear(obs_space + action_space, obs_space))

    def forward(self, obs, act):


        x = torch.cat([obs, act], dim=1)

        out = []

        for i in range(self.num_ensembles):
            out.append(self.ensembles[i].forward(x))

        std = torch.stack(out, dim=1).std()

        out = torch.stack(out, dim=0)

        return out, std