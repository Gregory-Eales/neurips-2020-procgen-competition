import ray
from ray.rllib.agents.pg.pg_tf_policy import post_process_advantages
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.framework import try_import_torch

import torch




def p2e_loss(policy, model, dist_class, train_batch):

    # get env data
    obs = train_batch[SampleBatch.CUR_OBS]
    next_obs = train_batch[SampleBatch.NEXT_OBS]

    reward = train_batch[SampleBatch.REWARDS]

    action = train_batch[SampleBatch.ACTIONS]
    act_prob = train_batch[SampleBatch.ACTION_PROB]
    log_prob = train_batch[SampleBatch.ACTION_LOGP]


    value_target = train_batch[Postprocessing.VALUE_TARGETS]
    prev_reward = train_batch[Postprocessing.ADVANTAGES]


    #
    ensemble_loss = torch.nn.MSELoss()(
        train_batch['ensemble_out'],
        torch.cat([])
        )

    #
    dynamics_loss = torch.nn.MSELoss()(
        train_batch['dynamics_out'],
        torch.cat([latent_state, action], dim=1)
        )

    #
    reconstruction = model.decoder(train_batch['encoder_out'])
    vae_loss = torch.nn.MSELoss()(reconstruction, obs)

    # 
    exp_loss = 

    return 


def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    ''' Return Dynamics, VAE, and Ensemble predictions '''
    return {
    'dynamics_out':model.dynamics_out,
    'ensemble_out':model.ensemble_out,
    'encoder_out':model.encoder_out
    }