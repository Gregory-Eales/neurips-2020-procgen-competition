import torch
from torch import nn
import ray
from ray.rllib.models.torch.recurrent_torch_model import RecurrentTorchModel
from ray.rllib.utils.annotations import override
import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch 

class RNNSubModule(RecurrentTorchModel):

        def __init__(self, obs_space, num_outputs):
            self.obs_size = _get_size(obs_space)
            self.rnn_hidden_dim = model_config["lstm_cell_size"]
            self.fc1 = nn.Linear(self.obs_size, self.rnn_hidden_dim)
            self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
            self.fc2 = nn.Linear(self.rnn_hidden_dim, num_outputs)
            self.value_branch = nn.Linear(self.rnn_hidden_dim, 1)
            self._cur_value = None

        @override(ModelV2)
        def get_initial_state(self):
            # make hidden states on same device as model
            h = [self.fc1.weight.new(
                1, self.rnn_hidden_dim).zero_().squeeze(0)]
            return h

        @override(ModelV2)
        def value_function(self):
            assert self._cur_value is not None, "must call forward() first"
            return self._cur_value

        @override(RecurrentTorchModel)
        def forward_rnn(self, input_dict, state, seq_lens):
            x = nn.functional.relu(self.fc1(input_dict["obs_flat"].float()))
            h_in = state[0].reshape(-1, self.rnn_hidden_dim)
            h = self.rnn(x, h_in)
            q = self.fc2(h)
            self._cur_value = self.value_branch(h).squeeze(1)
            return q, [h]