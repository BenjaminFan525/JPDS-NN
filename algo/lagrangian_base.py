import abc
import torch
import torch.optim as optim

class Lagrangian(abc.ABC):
    """ 
        Abstract base class for Lagrangian-base Algorithm
    """
    def __init__(self,
                 lagrangian_multiplier_init: float,
                 lambda_lr: float,
                 lambda_optimizer: str,
                 constraint: list = [],
                 **kwargs
                 ):
        self.lambda_lr = lambda_lr

        init_value = max(lagrangian_multiplier_init, 1e-5)
        self.lagrangian_multipliers = {
            key: torch.nn.Parameter(
                    torch.as_tensor(init_value, device=kwargs['device'] if 'device' in kwargs else 'cpu'),
                    requires_grad=True) 
            for key in constraint
        }
        self.lambda_range_projection = torch.nn.ReLU()
        # fetch optimizer from PyTorch optimizer package
        assert hasattr(optim, lambda_optimizer), \
            f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(optim, lambda_optimizer)
        self.lambda_optimizer = torch_opt(list(self.lagrangian_multipliers.values()), lr=lambda_lr)

    # def compute_lambda_loss(self, mean_ep_cost: torch.Tensor, cost_limit: torch.Tensor):
    #     """Penalty loss for Lagrange multiplier."""
    #     return -self.lagrangian_multiplier * (mean_ep_cost - cost_limit).mean()

    def update_lagrange_multiplier(self, ep_costs, cost_limits):
        """ Update Lagrange multiplier (lambda)
            Note: ep_costs obtained from: self.logger.get_stats('EpCosts')[0]
            are already averaged across MPI processes.
        """
        self.lambda_optimizer.zero_grad()
        lambda_loss = 0
        for key in self.lagrangian_multipliers.keys():
            lambda_loss -= self.lagrangian_multipliers[key] * (ep_costs[key] - cost_limits[key]).mean()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        for param in self.lagrangian_multipliers.values():
            param.data.clamp_(0)  # enforce: lambda in [0, inf]