from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch
from ray.rllib.models.modelv2 import ModelV2

import logging
from ray.tune.logger import Logger, UnifiedLogger
logger = logging.getLogger(__name__)

from collections import OrderedDict
import gym

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI, PublicAPI

import time




torch, nn = try_import_torch()


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=8, stride=1)


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

        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=8, stride=1)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.deconv4 = nn.ConvTranspose2d(128, 3, kernel_size=2, stride=2)

    def forward(self, x):

        out = x

        out = self.deconv1(out)
        out = nn.functional.relu(out)
        out = self.deconv2(out)
        out = nn.functional.relu(out)
        out = self.deconv3(out)
        out = nn.functional.relu(out)
        out = self.deconv4(out)
        out = nn.functional.relu(out)

        return out


class PolicyHead(nn.Module):

    def __init__(self):

        pass

class ValueHead(nn.Module):

    def __init__(self):

        pass

class TransitionModel(nn.Module):

    def __init__(self):

        pass




class MasterModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc(x)
        x = nn.functional.relu(x)
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        self._value = value.squeeze(1)
        return logits, state

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

    x = torch.ones(1, 3, 64, 64)

    print(x.shape)

    encoder = Encoder()
    decoder = Decoder()

    y = encoder.forward(x)

    print(y.shape)

    y = decoder.forward(y)

    print(y.shape)



