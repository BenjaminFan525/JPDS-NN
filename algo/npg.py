import torch
from algo.policy_gradiant import PG
from utils import get_flat_params_from, conjugate_gradients, get_flat_gradients_from,set_param_values_to_model, sample_obs
import numpy as np

class NPG(PG):
    def __init__(
            self,
            algo: str = 'npg',
            cg_damping: float = 0.1,
            cg_iters: int = 10,
            target_kl: float = 0.01,
            **kwargs
    ):
        PG.__init__(
            self,
            algo=algo,
            target_kl=target_kl,
            use_linear_lr_decay=False,
            **kwargs)
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.target_kl = target_kl
        self.fvp_obs = None
        self.scheduler = None  

    def search_step_size(self,
                              step_dir,
                              g_flat,
                              p_dist,
                              data):
        """ 
            NPG use full step_size
        """
        accept_step = 1
        return step_dir, accept_step

    def algorithm_specific_logs(self):
        self.logger.log_tabular('Misc/AcceptanceStep')
        self.logger.log_tabular('Misc/Alpha')
        self.logger.log_tabular('Misc/FinalStepNorm')
        self.logger.log_tabular('Misc/gradient_norm')
        self.logger.log_tabular('Misc/xHx')
        self.logger.log_tabular('Misc/H_inv_g')

    def Fvp(self, p):
        """ 
            Build the Hessian-vector product based on an approximation of the KL-divergence.
            For details see John Schulman's PhD thesis (pp. 40) http://joschu.net/docs/thesis.pdf
        """
        self.ac.zero_grad()
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False):
            q_dist = self.ac(**self.fvp_obs, dist_only=True)
        with torch.no_grad():
            p_dist = self.ac(**self.fvp_obs, dist_only=True)
        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

        if self.ac.encoder.nodes_encoder.frozen_gnn:
            grads = torch.autograd.grad(kl, self.ac.actor_param_without_gnn.parameters(), create_graph=True)
        else:
            grads = torch.autograd.grad(kl, self.ac.actor_param.parameters(), create_graph=True)
            
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_p = (flat_grad_kl * p).sum()
        if self.ac.encoder.nodes_encoder.frozen_gnn:
            grads = torch.autograd.grad(kl_p, self.ac.actor_param_without_gnn.parameters(),
                                        retain_graph=False)
        else:
            grads = torch.autograd.grad(kl_p, self.ac.actor_param.parameters(),
                                        retain_graph=False)
        # contiguous indicating, if the memory is contiguously stored or not
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1)
                                       for grad in grads])

        return flat_grad_grad_kl + p * self.cg_damping

    def update(self):
        """
            Update actor, critic, running statistics
        """
        data = self.buf.get()

        # sub-sampling accelerates calculations
        mb_indices = np.arange(self.batch_size)[::4]
        self.fvp_obs=sample_obs(data['obs'], mb_indices)
        self.fvp_obs.update({'chosen_idx': data['act']['chosen_idx'][mb_indices],
                            'chosen_entry': data['act']['chosen_entry'][mb_indices]})
        # Update Policy Network
        self.ac.eval()
        self.update_policy_net(data)
        self.ac.train()
        # Update Value Function
        self.update_value_net(data=data)

    def update_policy_net(self, data):
        # Get loss and info values before update
        if self.ac.encoder.nodes_encoder.frozen_gnn:
            theta_old = get_flat_params_from(self.ac.actor_param_without_gnn)
        else:
            theta_old = get_flat_params_from(self.ac.actor_param)
        self.ac.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        self.loss_pi_before = loss_pi.item()
        with torch.no_grad():
            p_dist = self.ac(**data['obs'], **data['act'], dist_only=True)
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        g_flat = get_flat_gradients_from(self.ac.actor_param)

        assert len(theta_old) == len(g_flat), f"theta dim {len(theta_old)} does not match with gradiant dim {len(g_flat)}"

        # # flip sign since policy_loss = -(ration * adv)
        # g_flat *= -1

        x = conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()
        # Note that xHx = g^T x, but calculating xHx is faster than g^T x
        xHx = torch.dot(x, self.Fvp(x))  # equivalent to : g^T x
        # xHx = torch.dot(x, g_flat)
        assert xHx.item() >= 0, 'No negative values'

        # perform descent direction
        alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
        step_direction = alpha * x
        assert torch.isfinite(step_direction).all()

        # determine step direction and apply SGD step after grads where set
        # TRPO uses custom backtracking line search
        final_step_dir, accept_step = self.search_step_size(
            step_dir=step_direction,
            g_flat=g_flat,
            p_dist=p_dist,
            data=data,
        )

        # update actor network parameters
        new_theta = theta_old - final_step_dir
        if self.ac.encoder.nodes_encoder.frozen_gnn:
            set_param_values_to_model(self.ac.actor_param_without_gnn, new_theta)
        else:
            set_param_values_to_model(self.ac.actor_param, new_theta)

        with torch.no_grad():
            q_dist = self.ac(**data['obs'], **data['act'], dist_only=True)
            kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
            loss_pi, pi_info = self.compute_loss_pi(data=data)

        self.logger.store(**{
            'Values/Adv': data['adv'][self.obj].cpu().numpy().flatten(),
            'Entropy': pi_info['ent'],
            'KL': kl,
            'PolicyRatio': pi_info['ratio'],
            'Loss/Pi': self.loss_pi_before,
            'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
            'Misc/AcceptanceStep': accept_step,
            'Misc/Alpha': alpha.item(),
            'Misc/StopIter': 1,
            'Misc/FinalStepNorm': torch.norm(final_step_dir).cpu().numpy(),
            'Misc/xHx': xHx.item(),
            'Misc/gradient_norm': torch.norm(g_flat).cpu().numpy(),
            'Misc/H_inv_g': x.norm().item(),
        })