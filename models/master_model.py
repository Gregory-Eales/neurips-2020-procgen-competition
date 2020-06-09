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

        self.last_latent_obs = None


        self.vae_loss = torch.nn.MSELoss()

        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        obs = obs / 255.0  # scale to 0-1
        obs = obs.permute(0, 3, 1, 2)  # NHWC => NCHW
        
        latent_obs = self.encoder(obs)

        self.last_latent_obs = latent_obs

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


    @override(ModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        # Create a new input reader per worker.
        reader = JsonReader(self.input_files)
        input_ops = reader.tf_input_ops()

        # Define a secondary loss by building a graph copy with weight sharing.
        obs = restore_original_dimensions(
            tf.cast(input_ops["obs"], tf.float32), self.obs_space)
        logits, _ = self.forward({"obs": obs}, [], None)

        # You can also add self-supervised losses easily by referencing tensors
        # created during _build_layers_v2(). For example, an autoencoder-style
        # loss can be added as follows:
        ae_loss = self.vae_loss(loss_inputs["obs"], self.decoder(self.last_latent_obs))
   

        return policy_loss + ae_loss

    def custom_stats(self):
        return {
            "policy_loss": self.policy_loss,
            "vae_loss": self.imitation_loss,
            "ensemble_loss": self.imitation_loss,
            "dynamics_loss": self.imitation_loss
        }





# Register model in ModelCatalog
ModelCatalog.register_custom_model("master_model", MasterModel)


if __name__ == "__main__":

    pass



