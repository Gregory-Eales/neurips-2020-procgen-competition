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


from models.modules.auto_encoder import Encoder, Decoder


torch, nn = try_import_torch()


class PolicyHead(nn.Module):

    def __init__(self, num_outputs):

        super(PolicyHead, self).__init__()

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, num_outputs)

    def forward(self, x):

        out = x

        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        #out = nn.functional.relu(out)

        return out


class ValueHead(nn.Module):

    def __init__(self):

        super(ValueHead, self).__init__()

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):

        out = x

        out = self.fc1(out)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        #out = nn.functional.relu(out)

        return out

class TransitionModel(nn.Module):

    def __init__(self):

        super(TransitionModel, self).__init__()

        pass




class MasterModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        h, w, c = obs_space.shape
        shape = (c, h, w)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.policy_head = PolicyHead(num_outputs)
        self.value_head = ValueHead()

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

        policy = self.policy_head(latent_obs.reshape(-1, 1024))
        value = self.value_head(latent_obs.reshape(-1, 1024))
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

    @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        return policy_loss

# Register model in ModelCatalog
ModelCatalog.register_custom_model("master_model", MasterModel)



if __name__ == "__main__":

    from ray.rllib.contrib import alpha_zero

    x = torch.ones(1, 3, 64, 64)

    print(x.shape)

    encoder = Encoder()
    decoder = Decoder()
    policy = PolicyHead(15)
    value = ValueHead()

    y = encoder.forward(x)

    print(y.shape)

    print(policy(y.reshape(-1, 64)).shape)
    print(value(y.reshape(-1, 64)).shape)

    print(y.shape)

    y = decoder.forward(y)

    print(y.shape)



