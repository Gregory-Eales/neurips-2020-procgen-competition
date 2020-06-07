from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from .master_policy import MasterPolicy

from ray.rllib.agents import Trainer


# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # No remote workers by default.
    "num_workers": 0,
    # Learning rate.
    "lr": 0.0004,
})
# __sphinx_doc_end__
# yapf: enable


def get_policy_class(config):
        return MasterPolicy


MasterAgent= build_trainer(
    name="custom/MasterAgent",
    default_config=DEFAULT_CONFIG,
    default_policy=MasterPolicy,
    get_policy_class=get_policy_class)
