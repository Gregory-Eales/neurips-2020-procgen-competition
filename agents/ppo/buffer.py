import torch
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset

class Buffer(Dataset):

    def __init__(self):

        # store actions
        self.act = []

        # store old actions
        self.old_act = []

        # store state
        self.obs = []

        # store reward
        self.rew = []

        # store advantage
        self.adv = []

        self.old_policy = None

    def store_observation(self, obs):
        self.obs.append(obs)

    def store_reward(self, rwrd):
        self.rew.append(rwrd)

    def store_action(self, act):
        self.act.append(act)

    def store_old_action(self, old_act):
        self.old_act.append(old_act)

    def store_advantage(self, adv):
        self.adv.append(adv)

    def clear_buffer(self):
        # store actions
        self.act = []

        # store old actions
        self.old_act = []

        # store state
        self.obs = []

        # store reward
        self.rew = []

        # store advantage
        self.adv = []

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):

        obs = torch.Tensor(self.obs).reshape(-1, self.obs_dim)[idx]
        act = torch.Tensor(self.act).reshape(-1, self.act_dim)[idx]
        r = torch.Tensor(self.r).reshape(-1, 1)[idx]
        s_p = torch.Tensor(self.s_p).reshape(-1, self.obs_dim)[idx]
        t = torch.Tensor(self.t).reshape(-1, 1)[idx]

        return s, a, r, n_s, t

    def __iter__(self):

        s = torch.Tensor(self.s).reshape(-1, self.obs_dim)
        a = torch.Tensor(self.a).reshape(-1, self.act_dim)
        r = torch.Tensor(self.r).reshape(-1, 1)
        s_p = torch.Tensor(self.s_p).reshape(-1, self.obs_dim)
        t = torch.Tensor(self.t).reshape(-1, 1)
        
        num_batch = s.shape[0]//self.hparams.batch_size
        rem_batch = s.shape[0]%self.hparams.batch_size
        
        for i in range(num_batch):
            i1, i2 = i*self.hparams.batch_size, (i+1)*self.hparams.batch_size
        
            yield s[i1:i2], a[i1:i2], r[i1:i2], s_p[i1:i2], t[i1:i2]
        
        
        i1, i2 = -rem_batch, 0
        yield s[i1:i2], a[i1:i2], r[i1:i2], s_p[i1:i2], t[i1:i2]