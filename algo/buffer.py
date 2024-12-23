import torch
import numpy as np
from copy import deepcopy
from algo.core import discount_cumsum

class Buffer:
    def __init__(self,
                 gamma: float,
                 lam: float,
                 adv_estimation_method: str,
                 ):
        """
            A buffer for storing trajectories experienced by an agent interacting
            with the environment, and using Generalized Advantage Estimation (GAE)
            for calculating the advantages of state-action pairs.

            Important Note: Buffer collects only raw data received from environment.
        """
        self.gamma = gamma
        self.lam = lam
        self.adv_estimation_method = adv_estimation_method

        assert adv_estimation_method in ['gae', 'vtrace', 'plain']

    def calculate_adv_and_value_targets(self, vals, rews, lam=None):
        """ Compute the estimated advantage"""

        if self.adv_estimation_method == 'gae':
            # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
            lam = self.lam if lam is None else lam
            deltas = rews[:, :-1] + self.gamma * vals[:, 1:] - vals[:, :-1]
            numpy_adv = discount_cumsum(deltas.cpu().numpy(), self.gamma * lam)
            adv = torch.tensor(numpy_adv, device=vals.device)
            value_net_targets = adv + vals[:, :-1]

        elif self.adv_estimation_method == 'plain':
            # A(x, u) = Q(x, u) - V(x) = r(x, u) + gamma V(x+1) - V(x)
            adv = rews[:, :-1] + self.gamma * vals[:, 1:] - vals[:, :-1]

            # compute rewards-to-go, to be targets for the value function update
            # value_net_targets are just the discounted returns
            value_net_targets = torch.Tensor(discount_cumsum(rews.cpu().numpy(), self.gamma), device=vals.device)

        else:
            raise NotImplementedError

        return adv, value_net_targets

    def store(self, data, info, index, entry, cost, val, val_mask, prob, cumulated_cost, dist, cost_lim: dict=None):
        """
        Append one timestep of agent-environment interaction to the buffer.

        Important Note: Store only raw data received from environment!!!
        Note: perform reward scaling if enabled
        """
        self.obs_buf = {'data': data, 'info': info}
        self.act_buf = {'chosen_idx': index, 'chosen_entry': entry}

        self.cost_buf = cost
        self.val_buf = val
        self.val_mask_buf = val_mask
        self.logp_buf = torch.log(prob)
        self.cumulated_cost_buf = cumulated_cost

        self.adv_buf = {}
        self.target_val_buf = {}
        for key, value in self.val_buf.items():
            adv, target_val = self.calculate_adv_and_value_targets(value, self.cost_buf[key])
            self.adv_buf.update({key: adv})
            self.target_val_buf.update({key: target_val})

        self.cost_lim = cost_lim
        self.safety_state_buf = {}
        if cost_lim is not None:
            for key, value in cost_lim.items():
                numpy_cum_cost = discount_cumsum(self.cost_buf[key].cpu().numpy()[:, ::-1], 1)[:, ::-1].copy()
                self.safety_state_buf.update({
                    key: (value - torch.tensor(numpy_cum_cost, device=value.device)) / value})

        self.dist = dist
    
    def fusing_s(self, obj, factor:float=1):
        self.adv_buf[obj] += factor * self.adv_buf['s']

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        data = dict(
            obs=self.obs_buf, 
            act=self.act_buf, 
            target_v=self.target_val_buf,
            adv=self.adv_buf, 
            log_p=self.logp_buf,
            cost=self.cost_buf,
            cost_val = self.val_buf,
            val_mask = self.val_mask_buf,
            cum_c = self.cumulated_cost_buf,
            cost_limit = self.cost_lim,
            safety_state=self.safety_state_buf,
            p_dist = self.dist
        )

        return data
