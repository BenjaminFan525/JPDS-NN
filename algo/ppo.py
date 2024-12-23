import torch
from algo.policy_gradiant import PG

class PPO(PG):
    def __init__(
            self,
            algo: str = 'ppo',
            clip: float = 0.2,
            **kwargs
    ):
        super().__init__(algo=algo, **kwargs)
        self.clip = clip

    def compute_loss_pi(self, data: dict):
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

        # Useful extra info
        ent = dist.entropy().mean().item()
        pi_info = dict(ent=ent, ratio=ratio_clip[val_mask[:, :-1]].mean().item())
        return loss_pi, pi_info