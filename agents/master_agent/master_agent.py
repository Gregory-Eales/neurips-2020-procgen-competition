import torch
import pytorch_lightning as pl

from policy_network import PolicyNetwork
from value_network import ValueNetwork

class MasterAgent(pl.LightningModule):


	def __init__(self, hparams):

		self.hparams = hparams

		self.policy_network = PolicyNetwork(hparams)
		self.value_network = ValueNetwork(hparams)
		self.q_network = QNetwork(hparams)
		self.model_networ = ModelNetwork(hparams)


		# trainer
        self.trainer = pl.Trainer(gpus=self.hparams.gpu, max_epochs=self.hparams.num_epochs)


	def act(self, s):

		s = torch.tensor(s).reshape(-1, 3, 64, 64).float()
		prediction = self.policy_network.forward(s)
		action_probabilities = torch.distributions.Categorical(prediction)
		action = action_probabilities.sample().item()
		log_prob = (action_probabilities.probs[0][action]).reshape(1, 1)

		# STORE LOG PROBS

		return action

	def calculate_advantages(self):
		pass

	def forward(self, x):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        s, a, r, s_p, t = batch
   
        # train policy
        if optimizer_idx == 0:
            return None

        # train value
        if optimizer_idx == 1:
            return None

    def configure_optimizers(self):
        lr = self.hparams.lr
        
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

   


