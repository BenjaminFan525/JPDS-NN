import torch
from algo.policy_gradiant import PG
# from safepo.common.core import ConstrainedPolicyGradientAlgorithm
from algo.lagrangian_base import Lagrangian
import torch.optim as optim
import numpy as np
from copy import deepcopy

class ESB_PPO_Lag_ADV(PG,Lagrangian):
    '''
    
    '''
    def __init__(
            self,
            algo='esb_ppo-lag_adv', 
            clip=0.2, 
            alpha_limit: float = 0.001,
            lambda_init: float = 0.001,
            lagrangian_multiplier_init=0.001,
            lambda_lr=0.008, 
            alpha_lr=0.01,
            lambda_optimizer='Adam', 
            **kwargs
        ):
        PG.__init__(
            self, 
            algo=algo, 
            **kwargs
        )

        Lagrangian.__init__(
            self, 
            lagrangian_multiplier_init=lagrangian_multiplier_init, 
            lambda_lr=lambda_lr, 
            lambda_optimizer=lambda_optimizer,
            **kwargs
        )

        self.clip = clip
        self.alphas = {key: torch.tensor(1 - alpha_limit, device=kwargs['device'] if 'device' in kwargs else 'cpu') 
                       for key in self.constraint}
        self.alpha_limit = alpha_limit

        # Log Lagrangian init
        self.SCALE_alpha_MIN_MAX = (0, 1)
        # self.use_lagrangian_penalty = use_lagrangian_penalty

        init_value = max(lambda_init, 1e-5)
        self.log_lams = {
            key: torch.nn.Parameter(
                    torch.log(torch.tensor(init_value, device=kwargs['device'] if 'device' in kwargs else 'cpu')),   
                    requires_grad=True)
            for key in self.constraint
        }

        # fetch optimizer from PyTorch optimizer package
        assert hasattr(optim, lambda_optimizer), \
            f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(optim, lambda_optimizer)
        self.alpha_optimizer = torch_opt(list(self.log_lams.values()), lr=alpha_lr)

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        for key in self.constraint:
            self.logger.log_tabular('LagrangeMultiplier/' + key, 
                                    self.lagrangian_multipliers[key].data.item())
            self.logger.log_tabular('alpha/' + key,
                                    self.alphas[key].item())
            self.logger.log_tabular('ESB/Gap1/' + key, 
                                    np.mean(self.G1[key][2:]).item())
            self.logger.log_tabular('ESB/Gap2/' + key, 
                                    np.mean(self.G2[key][2:]).item())

    def compute_loss_cost_performance(self, data):
        """
        tips: 
        1 - epsilon : represents the degree of cost if ep_costs> cost_limit, 1-epsilon close to 0
        constraints improve
        ( 1 - self.buf.gamma ): a constant can be modified (adjust performance)
        a possible matching with saute rl is to replace (1 - epsilon) with a z_t's formulation 
        """       
        safety_states = data['safety_state']
        lae = {}
        B1 = {}
        B2 = {}

        for key in self.constraint:
            beta = 1 + torch.clip(torch.tanh(safety_states[key][:, 1:]), -1, 0)
            lae.update({key: (data['adv'][key])})

            B1.update({key: (1 - self.gamma) * data['cost_val'][key][:, 1:] - data['cost'][key][:, :-1]})
            B2.update({key: self.alphas[key] * (1 - beta) / (1 - self.alphas[key]) * data['cost_val'][key][:, 1:]})

        return lae, B1, B2

    def compute_loss_pi(self, data: dict):
        # Policy loss
        _, _, _, _, prob, dist, val_mask = self.ac(**data['obs'], **data['act'], criticize=False)
        _log_p = torch.log(prob)
        # Importance ratio
        ratio = torch.exp(_log_p - data['log_p'])
        ratio_clip = torch.clamp(ratio, 1-self.clip, 1+self.clip)
        loss_pi = (torch.max(ratio * data['adv'][self.obj], ratio_clip * data['adv'][self.obj]))[data['val_mask'][:, :-1]].mean()

        kl = torch.distributions.kl.kl_divergence(data['p_dist'], dist).mean().item()
        if kl > self.target_kl * 2.0:
            self.pi_lr = max(1e-5, self.pi_lr / 1.5)
        elif kl < self.target_kl/2.0 and kl > 0.0:
            self.pi_lr = min(1e-4, self.pi_lr * 1.5)
        for param_group in self.pi_optimizer.param_groups:
            param_group['lr'] = self.pi_lr

        # calculate cost penalties
        surrogate = deepcopy(data['cum_c'])
        for key in self.constraint:
            surrogate[key] = surrogate[key] + ((ratio_clip.detach() - 1) * self.lae[key])[val_mask[:, :-1]].mean() / (1 - self.gamma)
            self.G1[key].append((((ratio_clip - 1) * self.B1[key])[val_mask[:, :-1]].mean() / (1 - self.gamma)).item())
            self.G2[key].append((((ratio_clip - 1) * self.B2[key])[val_mask[:, :-1]].mean() / (1 - self.gamma)).item())
        self.update_lagrange_multiplier(surrogate, data['cost_limit'])
        # ensure that lagrange multiplier is positive
        penaltys = {key: self.lambda_range_projection(lam).item() for key, lam in self.lagrangian_multipliers.items()}
        lam_total = (1 + sum(penaltys.values()))

        loss_pi /= lam_total
        
        for key, lam in penaltys.items():
            l = lam / lam_total
            loss_pi += l * (torch.max(ratio * self.lae[key], ratio_clip * self.lae[key]))[data['val_mask'][:, :-1]].mean()

        # Useful extra info
        ent = dist.entropy().mean().item()
        pi_info = dict(ent=ent, ratio=ratio[val_mask[:, :-1]].mean().item())

        return loss_pi, pi_info

    def update_constraint_coeff(self, data):
        # update constrain coefficient
        lae, _, _ = self.compute_loss_cost_performance(data=data)
        with torch.no_grad():
            _, _, _, _, prob, dist, val_mask = self.ac(**data['obs'], **data['act'], criticize=False)
        _log_p = torch.log(prob)
        # Importance ratio
        ratio = torch.exp(_log_p - data['log_p'])

        loss_cost = {
            key: (ratio * lae[key])[val_mask[:, :-1]].mean()
            for key in self.constraint
        }
        # update Lagrange multiplier parameter
        self.update_alphas(loss_cost)
        self.lae, self.B1, self.B2 = self.compute_loss_cost_performance(data=data)
        self.G1 = {key: [] for key in self.constraint}
        self.G2 = {key: [] for key in self.constraint}
        
    def compute_alpha_loss(self, cost_constrain: dict):
        """Penalty loss for Lagrange multiplier."""
        loss = 0
        for key, val in cost_constrain.items():
            loss -= self.log_lams[key] * val
        return loss

    def update_alphas(self, loss_costs):
        """ 
        Update inner Lagrange multiplier (lam_alpha)
        """
        self.alpha_optimizer.zero_grad()
        alpha_loss = self.compute_alpha_loss(loss_costs)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # self.log_lagrangian_multiplier.data.clamp_(0)  # enforce: lambda in [0, inf]
        # tanh(k/x), smaller k, weaker constraint
        for key in self.constraint:
            self.alphas[key] = torch.clip(torch.tanh(0.05 / self.lambda_l_op(key).detach()), 0.01, 1 - self.alpha_limit)
    
    def lambda_l_op(self, key):
        '''
            return : lambda_l  using learned lambda parameter
        '''
        return torch.clamp(torch.exp(self.log_lams[key]), *self.SCALE_alpha_MIN_MAX)
    
    def update(self):
        data = self.buf.get()
        # First update Lagrange multiplier parameter
        self.ac.eval()
        self.update_constraint_coeff(data=data)
        # now update policy and value network
        if self.ac.sel_enc.seq_enc == 'gru' or self.ac.sel_enc.seq_enc == 'pure_gru' or self.ac.sel_enc.seq_enc == 'slice_gru':
            self.ac.sel_enc.train()
        self.update_policy_net(data=data)
        self.ac.train()
        self.update_value_net(data=data)