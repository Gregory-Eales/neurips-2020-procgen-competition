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


import random


if __name__ == "__main__":
    from modules.auto_encoders.mini_auto_encoder import MiniEncoder, MiniDecoder
    from modules.auto_encoders.large_auto_encoder import LargeEncoder, LargeDecoder
    from modules.auto_encoders.medium_auto_encoder import MediumEncoder, MediumDecoder
    from modules.auto_encoders.small_auto_encoder import SmallEncoder, SmallDecoder
    from modules.policy_head import PolicyHead
    from modules.value_head import ValueHead
    from modules.res_block import LinearResBlock
    from modules.ensemble import Ensemble
    from modules.dynamics_model import DynamicsModel
    from modules.rnn_submodule import RNNSubModule

else:
    from models.modules.auto_encoders.mini_auto_encoder import MiniEncoder, MiniDecoder
    from models.modules.auto_encoders.small_auto_encoder import SmallEncoder, SmallDecoder
    from models.modules.auto_encoders.medium_auto_encoder import MediumEncoder, MediumDecoder
    from models.modules.auto_encoders.large_auto_encoder import LargeEncoder, LargeDecoder
    from models.modules.policy_head import PolicyHead
    from models.modules.value_head import ValueHead
    from models.modules.res_block import LinearResBlock
    from models.modules.ensemble import Ensemble
    from models.modules.dynamics_model import DynamicsModel
    from models.modules.rnn_submodule import RNNSubModule


torch, nn = try_import_torch()


class P2EModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):

        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        nn.Module.__init__(self)

        h, w, c = obs_space.shape
        shape = (c, h, w)

        linear_num = 64
        latent_size = 64

        
        self.encoder = SmallEncoder()
        self.decoder = SmallDecoder()
        
        self.ensemble = Ensemble(
            num_ensembles=10,
            obs_space=latent_size,
            action_space=action_space
            )

        self.dynamics_model = DynamicsModel(
            action_size=action_space,
            latent_size=latent_size,
            hidden_size=64+1
            )

        self.policy_head = PolicyHead(linear_num, num_outputs)
        self.value_head = ValueHead(linear_num)


        self.fc = nn.Linear(256, linear_num)

        
        self.lb1 = LinearResBlock(linear_num)
        self.lb2 = LinearResBlock(linear_num)
        self.lb3 = LinearResBlock(linear_num)
        self.lb4 = LinearResBlock(linear_num)
        

        self.last_latent_obs = None


        self.vae_loss_fn = torch.nn.MSELoss()
        self.vae_loss = None

        self.count = 0



        
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        obs = obs / 255.0  # scale to 0-1
        obs = obs.permute(0, 3, 1, 2)  # NHWC => NCHW
        
        latent_obs = self.encoder(obs)

        ensemble_out, ensemble_disagreement = self.ensemble(latent_obs)

        l = self.fc(latent_obs.detach())
        l = nn.functional.leaky_relu(l)
        
        l = self.lb1(l)
        l = self.lb2(l)
        l = self.lb3(l)
        l = self.lb4(l)
        
        policy = self.policy_head(l)
        value = self.value_head(l)

        self._value = value.squeeze(1)
        self.encoder_out = latent_obs
        self.dynamics_out = dynamics_out
        self.ensemble_out = ensemble_out
        self.ensemble_disagreement = ensemble_disagreement

        next_obs, reward, terminal = self.dynamics_model(latent_obs, policy)

        return policy, state

    @override(ModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value


    """
    @override(ModelV2)
    def custom_loss(self, policy_loss, train_batch):

        # GET VAE LOSS
        if self.count % 10 == 0:

            obs = train_batch[SampleBatch.CUR_OBS]
            obs = obs / 255.0  # scale to 0-1
            obs = obs.permute(0, 3, 1, 2) 

            vae_loss = self.vae_loss_fn(obs, self.decoder(self.last_latent_obs))
            self.vae_loss = vae_loss


        # GET DYNAMICS LOSS

        dynamics_loss = self.dynamics_loss_fn()
        self.dynamics_loss = dynamics_loss
            

        # GET ENSAMBLE LOSS

        ensemble_loss = self.ensemble_loss_fn()
        self.ensemble_loss = ensemble_loss 

        self.count += 1
        return policy_loss + dynamics_loss + vae_loss + ensemble_loss 
    """


    def custom_stats(self):
        return {
            "policy_loss": self.policy_loss,
            "vae_loss": self.ae_loss,
            "ensemble_loss": self.ensemble_loss,
            "dynamics_loss": self.dynamics_loss
        }





# Register model in ModelCatalog
ModelCatalog.register_custom_model("p2e_model", P2EModel)


if __name__ == "__main__":

    model = P2EModel(torch.zeros(64, 64, 3), 15, 15, None, None)
    pp=0

    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    print(pp)