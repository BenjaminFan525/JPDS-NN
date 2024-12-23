import torch
from algo.policy_gradiant import PG
# from safepo.common.core import ConstrainedPolicyGradientAlgorithm
from algo.lagrangian_base import Lagrangian

class PPO_Lag(PG,Lagrangian):
    '''
    
    '''
    def __init__(
            self,
            algo='ppo-lag', 
            clip=0.2, 
            lagrangian_multiplier_init=0.001,
            lambda_lr=0.035, 
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

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        for key, val in self.lagrangian_multipliers.items():
            self.logger.log_tabular('LagrangeMultiplier/' + key,
                                val.data.item())


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

        # ensure that lagrange multiplier is positive
        penaltys = {key: self.lambda_range_projection(lam).item() for key, lam in self.lagrangian_multipliers.items()}
        lam_total = (1 + sum(penaltys.values()))

        loss_pi /= lam_total

        for key, lam in penaltys.items():
            l = lam / lam_total
            loss_pi += l * ((ratio * data['adv'][key]).mean())

        # Useful extra info
        ent = dist.entropy().mean().item()
        pi_info = dict(ent=ent, ratio=ratio[val_mask[:, :-1]].mean().item())

        return loss_pi, pi_info

    def update(self):
        data = self.buf.get()
        # First update Lagrange multiplier parameter
        self.update_lagrange_multiplier(data['cum_c'], data['cost_limit'])
        # now update policy and value network
        self.ac.eval()
        if self.ac.sel_enc.seq_enc == 'gru' or self.ac.sel_enc.seq_enc == 'pure_gru' or self.ac.sel_enc.seq_enc == 'slice_gru':
            self.ac.sel_enc.train()
        self.update_policy_net(data=data)
        self.ac.train()
        self.update_value_net(data=data)