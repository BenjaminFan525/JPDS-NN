from re import L
import numpy as np
import time
import torch
from utils.logger import EpochLogger
from model.ac import GNNAC
from utils import sample_obs
from algo.buffer import Buffer
import torch.optim as optim
from copy import deepcopy

class PG():
    def __init__(
            self,
            ac: GNNAC,
            logger_kwargs: dict,
            obj: str = 't',
            constraint: list = [],
            adv_estimation_method: str = 'gae',
            algo='pg',
            gamma: float = 0.99,
            lam: float = 0.95,  # GAE lambda
            batch_size: int = 32,
            max_grad_norm: float = 0.5,
            mini_batch: int = 8,  # used for value network training
            pi_lr: float = 3e-4,
            vf_lr: float = 1e-3,
            target_kl: float = 0.01,
            train_pi_iterations: int = 4,
            train_v_iterations: int = 4,
            use_discount_cost_update_lag=False,
            use_kl_early_stopping: bool = False,
            use_linear_lr_decay: bool = False,
            use_max_grad_norm: bool = True,
            seed: int = 0,
            total_epoch: int = 500,
            cur_epoch: int = 0,
            save_cfg: bool = True,
            **kwargs  # use to log parameters from child classes
    ):
        self.obj = obj
        self.constraint = constraint
        self.gamma = gamma # add because p3o
        self.adv_estimation_method = adv_estimation_method
        self.algo = algo

        self.lam = lam

        self.logger_kwargs = logger_kwargs

        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.mbs = mini_batch
        self.pi_lr = pi_lr

        self.target_kl = target_kl
        self.train_pi_iterations = train_pi_iterations
        self.train_v_iterations = train_v_iterations
        self.use_kl_early_stopping = use_kl_early_stopping # importance for ppo_base methods
        self.use_linear_lr_decay = use_linear_lr_decay
        self.use_max_grad_norm = use_max_grad_norm
        self.use_discount_cost_update_lag = use_discount_cost_update_lag

        # Set up logger and save configuration to disk
        # get local parameters before logger instance to avoid unnecessary print
        self.params = deepcopy(locals())
        # self.params.pop('self')
        self.params.pop('ac')
        # self.params.pop('env')
        # if 'kwargs' in self.params:
        #     self.params.update(**self.params.pop('kwargs'))
        self.logger = self._init_logger()
        if save_cfg:
            self.logger.save_config(self.params)

        # Set seed
        self.seed = seed
        # Setup actor-critic module
        self.ac = ac
        if ac.single_step:
            self.mbs = batch_size

        # Set up experience buffer
        self.buf = Buffer(
            gamma=gamma,
            lam=lam,
            adv_estimation_method=adv_estimation_method,
        )

        # Set up optimizers for policy and value function
        self.pi_optimizer = optim.Adam(self.ac.actor_param.parameters(), lr=pi_lr)
        self.vf_optimizer = optim.Adam(self.ac.critic_param.parameters(), lr=vf_lr)
        
        self.total_epoch = total_epoch
        self.epoch = cur_epoch
        # Set up scheduler for policy learning rate decay
        self.scheduler = self._init_learning_rate_scheduler()

        # # Set up model saving
        # self.logger.setup_torch_saver(self.ac)
        # self.logger.torch_save()

        # setup statistics
        self.loss_pi_before = 0.0
        self.loss_v_before = 0.0
        # self.logger.log('Start with training.')

    def _init_learning_rate_scheduler(self):
        scheduler = None
        if self.use_linear_lr_decay:
            import torch.optim
            def lm(epoch): return 1 - epoch / self.epochs  # linear anneal
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.pi_optimizer,
                lr_lambda=lm
            )
        return scheduler

    def _init_logger(self):
        self.params.pop('self')
        if 'kwargs' in self.params:
            self.params.update(**self.params.pop('kwargs'))
        logger = EpochLogger(**self.logger_kwargs)
        return logger

    def algorithm_specific_logs(self):
        """
            Use this method to collect log information.
            e.g. log lagrangian for lagrangian-base , log q, r, s, c for cpo, etc
        """
        pass
    
    def compute_loss_pi(self, data: dict):
        '''
            computing pi/actor loss

            Returns:
                torch.Tensor
        '''
        # Policy loss
        _, _, _, _, prob, dist, val_mask = self.ac(**data['obs'], **data['act'], criticize=False)
        _log_p = torch.log(prob)
        ratio = torch.exp(_log_p - data['log_p'])

        loss_pi = (ratio * data['adv'][self.obj])[data['val_mask'][:, :-1]].mean()

        # Useful extra info
        ent = dist.entropy().mean().item()
        pi_info = dict(ent=ent, ratio=ratio[val_mask[:, :-1]].mean().item())

        return loss_pi, pi_info

    def compute_loss_v(self, obs, act, ret):
        """
        computing value loss

        Returns:
            torch.Tensor
        """
        # tic = time.time()
        value, value_mask = self.ac(**obs, **act, actor_grad=False, criticize_only=True)
        # val = torch.cat([x[:, :-1] for x in value.values()])
        # val_mask = value_mask[:, :-1].repeat(len(value), 1)
        loss_v = 0
        loss_num = 0
        for key, val in value.items():
            loss_v = loss_v + ((val[:, :-1] - ret[key][:, :val.shape[1]-1]) ** 2)[value_mask[:, :-1]].mean()
            loss_num += 1
        # print('Computing loss V time: ' + str(time.time() - tic))
        return loss_v / loss_num

    def log(self, epoch: int):
        # Log info about epoch
        if self.scheduler and self.use_linear_lr_decay:
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()  # step the scheduler if provided
        else:
            current_lr = self.pi_lr

        self.logger.log_tabular('Iteration', epoch)
        self.logger.log_tabular('Values/Adv', min_and_max=True)
        self.logger.log_tabular('Loss/Pi', std=False)
        self.logger.log_tabular('Loss/Value')
        self.logger.log_tabular('Loss/DeltaPi')
        self.logger.log_tabular('Loss/DeltaValue')
        self.logger.log_tabular('Entropy')
        self.logger.log_tabular('KL')
        self.logger.log_tabular('Misc/StopIter')
        self.logger.log_tabular('Misc/Seed', self.seed)
        self.logger.log_tabular('PolicyRatio')
        self.logger.log_tabular('LR', current_lr)
        # some child classes may add information to logs
        self.algorithm_specific_logs()
        # self.logger.log_tabular('TotalEnvSteps', total_env_steps)

        self.logger.dump_tabular()

    def update(self):
        """
            Update actor, critic, running statistics
        """
        data = self.buf.get()
        
        # update actor
        # tic = time.time()
        self.ac.eval()
        if self.ac.sel_enc.seq_enc == 'gru' or self.ac.sel_enc.seq_enc == 'pure_gru' or self.ac.sel_enc.seq_enc == 'slice_gru':
            self.ac.sel_enc.train()
        self.update_policy_net(data=data)
        self.ac.train()
        # print('Updating Pi time: ' + str(time.time() - tic))

        # update critic
        # tic = time.time()
        self.update_value_net(data=data)
        # print('Updating V time: ' + str(time.time() - tic))
        
    def update_policy_net(self, data) -> None:
        # get prob. distribution before updates: used to measure KL distance
        # with torch.no_grad():
        #     self.p_dist = self.ac(**data['obs'], **data['act'], dist_only=True)
        # Get loss and info values before update
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        self.loss_pi_before = pi_l_old.item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iterations):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data=data)
            loss_pi.backward()
            if self.use_max_grad_norm:  # apply L2 norm
                torch.nn.utils.clip_grad_norm_(
                    self.ac.actor_param.parameters(),
                    self.max_grad_norm)

            self.pi_optimizer.step()
        
        with torch.no_grad():
            q_dist = self.ac(**data['obs'], **data['act'], dist_only=True)
            torch_kl = torch.distributions.kl.kl_divergence(
                data['p_dist'], q_dist).mean().item()
            # if self.use_kl_early_stopping:
            #     # average KL for consistent early stopping across processes
            #     if torch_kl > self.target_kl:
            #         self.logger.log(f'Reached ES criterion after {i+1} steps.')
            #         break

        # track when policy iteration is stopped; Log changes from update
        self.logger.store(**{
            'Loss/Pi': self.loss_pi_before,
            'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
            'Misc/StopIter': i + 1,
            'Values/Adv': data['adv'][self.obj].cpu().numpy().flatten(),
            'Entropy': pi_info['ent'],
            'KL': torch_kl,
            'PolicyRatio': pi_info['ratio']
        })

    def update_value_net(self, data: dict) -> None:
        loss_v = self.compute_loss_v(data['obs'], 
                                     data['act'], 
                                     data['target_v'])
        self.loss_v_before = loss_v.item()

        indices = np.arange(self.batch_size)
        val_losses = []
        for _ in range(self.train_v_iterations):
            np.random.shuffle(indices)  # shuffle for mini-batch updates
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.batch_size, self.mbs):
                end = start + self.mbs  # iterate mini batch times
                mb_indices = indices[start:end]
                self.vf_optimizer.zero_grad()
                
                loss_v = self.compute_loss_v(
                    obs=sample_obs(data['obs'], mb_indices),
                    act={'chosen_idx': data['act']['chosen_idx'][mb_indices],
                         'chosen_entry': data['act']['chosen_entry'][mb_indices]},
                    ret={key: x[mb_indices] for key, x in data['target_v'].items()})
                # tic = time.time()
                loss_v.backward()
                # print('Loss V time: ' + str(time.time() - tic))
                if self.use_max_grad_norm:  # apply L2 norm
                    torch.nn.utils.clip_grad_norm_(self.ac.critic_param.parameters(),
                                                    self.max_grad_norm)
                val_losses.append(loss_v.item())
                # average grads across MPI processes
                self.vf_optimizer.step()

        self.logger.store(**{
            'Loss/DeltaValue': np.mean(val_losses) - self.loss_v_before,
            'Loss/Value': self.loss_v_before,
        })