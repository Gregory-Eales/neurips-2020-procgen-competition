import torch
from tqdm import tqdm
import numpy as np

import pytorch_lightning as pl
from matplotlib import pyplot as plt

from policy_net import PolicyNetwork
from value_net import ValueNetwork
from buffer import Buffer

class PPO(pl.LightningModule):

	def __init__(self, hparams):

		self.hparams = hparams

		# store parameters
		self.lr = self.hparams.lr
		self.input_dims = self.hparams.in_dim
		self.output_dims = self.hparams.out_dim

		# initialize policy network
		self.policy_network = PolicyNetwork(hparams)

		# initialize old policy
		self.old_policy_network = PolicyNetwork(hparams)
		state_dict = self.policy_network.state_dict()
		self.old_policy_network.load_state_dict(state_dict)

		# initialize value network
		self.value_network = ValueNetwork(0.0001, in_dim, 1)

		# initialize vpg buffer
		self.buffer = Buffer()

	def act(self, s):

		# convert to torch tensor
		s = torch.tensor(s).reshape(-1, 3, 64, 64).float()

		# get policy prob distrabution
		prediction = self.policy_network.forward(s)

		# get action probabilities
		action_probabilities = torch.distributions.Categorical(prediction)

		# sample action
		action = action_probabilities.sample()
		action = action.item()
		# get log prob
		log_prob = (action_probabilities.probs[0][action]).reshape(1, 1)

		# get old prob
		old_p = self.old_policy_network.forward(s)
		old_ap = torch.distributions.Categorical(old_p)
		old_log_prob = (old_ap.probs[0][action]).reshape(1, 1)


		self.buffer.store_probs(log_prob, old_log_prob)

		return action

	def calculate_advantages(self, observation, prev_observation):

		observation = torch.from_numpy(observation).float()
		observation = torch.tensor(observation).reshape(-1, 3, 64, 64).float()
		prev_observation = torch.from_numpy(prev_observation).float()
		prev_observation = torch.tensor(prev_observation).reshape(-1, 3, 64, 64).float()

		# compute state value
		v = self.value_network.forward(prev_observation)

		# compute action function value
		q = self.value_network.forward(observation)

		# calculate advantage
		a = q - v + 1

		return a.detach().numpy()

	def env_step(self):

		for i_episode in range(episodes):

			s = env.reset()

			for t in range(steps):

				a = self..act(s, epsilon=epsilon)

				s_p, r, d, info = env.step(a)
				if t==steps-1: d = True

				self.store(s, a, r, s_p, d)
				s = s_p

				if d:
					break

	def training_step(self, batch, batch_idx, optimizer_idx):
        s, a, r, s_p, t = batch
   
        # train policy
        if optimizer_idx == 0:
            return self.policy_network.loss(r, adv)

        # train value
        if optimizer_idx == 1:
            return self.value_network.loss(input, target)

    def train_dataloader(self):
        return DataLoader(self.buffer, batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        lr = self.hparams.lr
        
        opt_p = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_v = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_p, opt_v], []
			


def main():

	import gym

	torch.manual_seed(1)
	np.random.seed(1)

	env = gym.make("procgen:procgen-{}-v0".format("coinrun"))
	vpg = PPO(alpha=0.001, in_dim=64, out_dim=15)

	vpg.train(env, n_epoch=2, n_steps=200, render=False, verbos=False)

if __name__ == "__main__":
	main()