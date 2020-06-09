from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
from ray.rllib.models.modelv2 import ModelV2


from collections import OrderedDict
import gym

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI

import time


from models.modules.large_auto_encoder import LargeEncoder, LargeDecoder
from models.modules.policy_head import PolicyHead
from models.modules.value_head import ValueHead
from models.modules.res_block import LinearResBlock
from models.modules.ensemble import Ensemble
from models.modules.dynamics_model import DynamicsModel


torch, nn = try_import_torch()


class MasterModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        h, w, c = obs_space.shape
        shape = (c, h, w)

        linear_num = 0

        self.encoder = LargeEncoder()
        self.decoder = LargeDecoder()

        self.policy_head = PolicyHead(64, num_outputs)
        self.value_head = ValueHead(64)


        self.fc = nn.Linear(1024, 64)


        self.lb1 = LinearResBlock(64)
        self.lb2 = LinearResBlock(64)
        self.lb3 = LinearResBlock(64)
        self.lb4 = LinearResBlock(64)

        """
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        """
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        obs = obs / 255.0  # scale to 0-1
        obs = obs.permute(0, 3, 1, 2)  # NHWC => NCHW
        
        latent_obs = self.encoder(obs)
        #decoder_input = latent_obs.reshape(-1, 256*2*2, 1, 1)
        #reconstruction = self.decoder(decoder_input)

        l = self.fc(latent_obs)
        l = self.lb1(l)
        l = self.lb2(l)
        l = self.lb3(l)
        l = self.lb4(l)

        policy = self.policy_head(l)
        value = self.value_head(l)
        self._value = value.squeeze(1)

        #next_obs, reward, terminal = self.world_model(latent_obs, policy)

        return policy, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value

    @override(TorchModelV2)
    def from_batch(self, train_batch, is_training=True):
        """Convenience function that calls this model with a tensor batch.
        All this does is unpack the tensor batch to call this model with the
        right input dict, state, and seq len arguments.
        """

        input_dict = {
            "obs": train_batch[SampleBatch.CUR_OBS],
            "is_training": is_training,
        }
        if SampleBatch.PREV_ACTIONS in train_batch:
            input_dict["prev_actions"] = train_batch[SampleBatch.PREV_ACTIONS]
        if SampleBatch.PREV_REWARDS in train_batch:
            input_dict["prev_rewards"] = train_batch[SampleBatch.PREV_REWARDS]
        states = []
        i = 0
        while "state_in_{}".format(i) in train_batch:
            states.append(train_batch["state_in_{}".format(i)])
            i += 1
        return self.__call__(input_dict, states, train_batch.get("seq_lens"))





# Register model in ModelCatalog
ModelCatalog.register_custom_model("master_model", MasterModel)


if __name__ == "__main__":

    pass



