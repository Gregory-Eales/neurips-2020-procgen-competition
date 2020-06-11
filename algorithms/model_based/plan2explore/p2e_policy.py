import ray
from ray.rllib.agents.pg.pg_tf_policy import post_process_advantages
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.framework import try_import_torch






def extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    ''' Return Dynamics, VAE, and Ensemble predictions '''
    return {'dynamics_out':model.dynamics_out, 'ensemble_out':model.ensemble_out, 'vae_out'}