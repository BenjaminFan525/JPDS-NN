import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from model.ac import GNNAC
from data.dataset import RLDataset
from data.dataloader import RLDataLoader
import yaml
from utils import dense_fit, fit, decode
from algo.buffer import Buffer
import matplotlib.pyplot as plt
import json
from algo import REGISTRY
from utils.logger import setup_logger_kwargs
from algo import PG
from env import arrangeSimulator
from copy import deepcopy
from tqdm import tqdm

def load_model(checkpoint, device):
    cfg = os.path.join('/'.join(checkpoint.split('/')[:-1]), 'config.json')
    with open(cfg, 'r') as f:
        cfg = json.load(f)
    
    ac_cfg = cfg['ac']

    ac = GNNAC(**ac_cfg, device=device)
    checkpoint = torch.load(checkpoint, map_location=device)
    ac.load_state_dict(checkpoint['model'])  
    ac.tau = cfg['anneal_final'] + (1 - cfg['anneal_final']) * (1 - checkpoint['epoch'] / cfg['epochs'])
    return ac

def validate(data_loader, model: PG, save_dir='.', render=False):
    """Used to monitor progress on a validation set & optionally plot solution."""
    # tic = time.time()
    model.ac.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    _, ax = plt.subplots()
    ax.axis('equal')

    r_final = []
    tqdm_data_loader = tqdm(data_loader)
    tqdm_data_loader.set_description('Validation')
    for batch_idx, batch in enumerate(tqdm_data_loader):

        if render:
            bsz_data, info, fields, car_cfgs, _ = batch
            field_matrices = [[field.D_matrix, field.ori, field.des, field.line_length] for field in fields]
        else:
            bsz_data, info, field_matrices, car_cfgs, _ = batch


        for key, _ in bsz_data.items():
            bsz_data[key] = bsz_data[key].to(device)
        for key, _ in info.items():
            info[key] = info[key].to(device)
        
        # Full forward pass through the dataset
        with torch.no_grad():
            seq_enc, _, _, _, _, _, _ = model.ac(bsz_data, info, deterministic=True, criticize=False)

        # Get rewards
        # for item in seq_enc:
        #     print(item)
        Ts = list(map(decode, seq_enc)) # decode the arrangements

        for bidx, (T, field_matrix, car_cfg) in enumerate(zip(Ts, field_matrices, car_cfgs)):
            r_final.append(fit(field_matrix[0], 
                    field_matrix[1], 
                    field_matrix[2],
                    car_cfg, 
                    field_matrix[3],
                    T, 
                    tS_t=False,
                    type='all'))
            if bidx == 0:
                s, t, c = r_final[-1]

        if render:
            name = 'batch%d.png'%(batch_idx)
            path = os.path.join(save_dir, name)
            sim = arrangeSimulator(fields[0], car_cfgs[0])
            sim.init_simulation(Ts[0], init_simulator=False)
            sim.render_arrange(ax)
            plt.title(f'$s_P$={np.round(s, 2)}m, $t_P$={np.round(t, 2)}s, $c_P$={np.round(c, 2)}L', fontsize=20)
            plt.savefig(path)
            ax.clear()

    plt.close('all')
    model.ac.train()
    r_final = np.array(r_final)
    mean_r = np.mean(r_final, axis=0)
    stc = {
        's': mean_r[0],
        't': mean_r[1],
        'c': mean_r[2]
    }
    # print('Validating time: ' + str(time.time() - tic))
    return stc

def train(model: PG, save_dir, train_data, valid_data, obj, 
          reward_coef, batch_size, epochs, cur_epoch=0, total_iter=0,
          valid_every=10, gnn_freeze_epochs=0, anneal=False, fusing_s=False, 
          fusing_epoch=4, final_reward=False, total_time=False, 
          total_time_epoch=4, anneal_final=0.1, constraint=[], cost_coef={}, 
          cost_val_limits={}, **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_loader = RLDataLoader(dataset=train_data, batch_size=batch_size, 
                                        shuffle=True, num_workers=0, drop_last=True)
    valid_loader = RLDataLoader(dataset=valid_data, batch_size=30, shuffle=False)

    best_reward = np.inf
    best_cost = {}
    if constraint is None:
        constraint = []
    for constraint_key in constraint:
        best_cost.update({constraint_key: np.inf})

    while cur_epoch < epochs:

        model.ac.train()

        times  = []
        cum_r = {
            's': [],
            't': [],
            'c': [],
            }

        epoch_start = time.time()
        start = epoch_start

        if cur_epoch < gnn_freeze_epochs:
            model.ac.encoder.freeze_gnn()
        else:
            model.ac.encoder.unfreeze_gnn()
        
        tqdm_data_loader = tqdm(train_loader)
        for batch_idx, batch in enumerate(tqdm_data_loader):
            if total_iter % valid_every == 0:
                # Save rendering of validation set tours
                valid_dir = os.path.join(save_dir, 'valid', str(cur_epoch), str(total_iter))

                valid_loader.dataset.use_field = True
                stc = validate(valid_loader, model, valid_dir, render=True)
                valid_loader.dataset.use_field = False

                with open(os.path.join(save_dir, 'log.txt'), 'a') as f:
                    f.write(str(cur_epoch) + ' ' + str(total_iter) + ' ' + \
                            str(total_iter * batch_size) + ' ' + \
                            str(stc['s']) + ' ' + str(stc['t']) + ' ' + str(stc['c']) + '\n')

                cost_satisfied = True
                for constraint_key in constraint:
                    if stc[constraint_key] > cost_val_limits[constraint_key] * cost_coef[constraint_key]:
                        cost_satisfied = False

                # Save best model parameters
                if (stc[obj] < best_reward) and cost_satisfied:
                    best_reward = stc[obj]
                    save_path = os.path.join(save_dir, 'best_model' + str(cur_epoch) + '.pt')
                    checkpoint = {
                        'epoch': cur_epoch,
                        'model': model.ac.state_dict(),
                        'actor_optim': model.pi_optimizer.state_dict(),
                        'critic_optim': model.vf_optimizer.state_dict(),
                        # 'lagrangmdvrpn_multiplier': lagrangmdvrpn_multiplier
                    }
                    if hasattr(model, 'lagrangmdvrpn_multipliers'):
                        checkpoint.update({
                            'lagrangmdvrpn_multiplier': model.lagrangmdvrpn_multipliers,
                            'lagrangmdvrpn_optimizer': model.lambda_optimizer.state_dict()
                        })
                    torch.save(checkpoint, save_path)
                for key, val in best_cost.items():
                    if stc[key] < val:
                        best_cost[key] = stc[key]
                        save_path = os.path.join(save_dir, 'best_model' + str(cur_epoch) + '_' + key + '.pt')
                        checkpoint = {
                            'epoch': cur_epoch,
                            'model': model.ac.state_dict(),
                            'actor_optim': model.pi_optimizer.state_dict(),
                            'critic_optim': model.vf_optimizer.state_dict(),
                            # 'lagrangmdvrpn_multiplier': lagrangmdvrpn_multiplier
                        }
                        if hasattr(model, 'lagrangmdvrpn_multipliers'):
                            checkpoint.update({
                                'lagrangmdvrpn_multiplier': model.lagrangmdvrpn_multipliers,
                                'lagrangmdvrpn_optimizer': model.lambda_optimizer.state_dict()
                            })
                        torch.save(checkpoint, save_path)

            total_iter += 1
            # tic = time.time()
            bsz_data, info, field_matrices, car_cfgs, cost_limits = batch

            for key, _ in bsz_data.items():
                bsz_data[key] = bsz_data[key].to(device)
            for key, _ in info.items():
                info[key] = info[key].to(device)
            for key, _ in cost_limits.items():
                cost_limits[key] = cost_limits[key].to(device)
            
            # Full forward pass through the dataset
            with torch.no_grad():
                model.ac.eval()
                seq_enc, choice, index, entry, prob, dist, value, value_mask = model.ac(bsz_data, info)
                model.ac.train()
            # Get rewards
            Ts = list(map(decode, seq_enc)) # decode the arrangements
            r = {'s': [], 't': [], 'c': []}
            cost = {}
            for key, val in value.items():
                cost.update({key: torch.zeros_like(val)})

            if cur_epoch >= total_time_epoch:
                total_time = False
            for T, field_martix, car_cfg in zip(Ts, field_matrices, car_cfgs):
                ret = dense_fit(field_martix[0], 
                            field_martix[1], 
                            field_martix[2],
                            car_cfg, 
                            field_martix[3],
                            T,
                            total_time)
                for key in r.keys():
                    r[key].append(ret[key])

            cur_cum_r = {}
            for key in r.keys():
                r[key] = pad_sequence(r[key], batch_first=True)
                cum_key = torch.sum(r[key], dim=-1, keepdim=True)
                cur_cum_r.update({key: deepcopy(cum_key)})
                cum_r[key].append(torch.mean(cum_key).item())
                r[key] *= reward_coef[key]
                cur_cum_r[key] *= reward_coef[key]
                cur_cum_r[key] = cur_cum_r[key].to(cost[key].device)

            for key in cost.keys():
                if model.ac.single_step:
                    cost[key][:, 0:1] = cur_cum_r[key]
                elif final_reward:
                    cost[key][:, -2:-1] = cur_cum_r[key]
                else: 
                    length = r[key].shape[-1]
                    cost[key][:, :length] = r[key].to(cost[key].device)
            if fusing_s and obj != 's':
                if cur_epoch < fusing_epoch:
                    cost[obj] += 1 * cost['s']
                    cur_cum_r[obj] += 1 * cur_cum_r['s']
                elif cur_epoch == fusing_epoch:
                    cost[obj] += 0.5 * cost['s']
                    cur_cum_r[obj] += 0.5 * cur_cum_r['s']
            
            for key in cost_limits.keys():
                cost_limits[key] *= reward_coef[key] * cost_coef[key]

            model.buf.store(bsz_data, info, index, entry, cost, value, value_mask, prob, cur_cum_r, dist, cost_limits)
            # print('Rollout time: ' + str(time.time() - tic))
            # if fusing_s and obj != 's' and cur_epoch < fusing_epoch:
            #     model.buf.fusing_s(obj, 1)
            # tic = time.time()

            model.update()
            # print('Updating time: ' + str(time.time() - tic)

            tqdm_data_loader.set_description(f'[Epoch {cur_epoch+1}/{epochs}] Reward {np.mean(cum_r[obj]):.4f}')

            # Log and store information
            if (total_iter - 1) % valid_every == 0:
                model.log(total_iter * batch_size)

        mean_reward = np.mean(cum_r[obj])

        # Save the weights      
        save_path = os.path.join(checkpoint_dir, 'checkpoint_Epoch' + str(cur_epoch) + '.pt')
        checkpoint = {
            'epoch': cur_epoch,
            'model': model.ac.state_dict(),
            'actor_optim': model.pi_optimizer.state_dict(),
            'critic_optim': model.vf_optimizer.state_dict(),
            # 'lagrangmdvrpn_multiplier': lagrangmdvrpn_multiplier
        }
        if hasattr(model, 'lagrangmdvrpn_multipliers'):
            checkpoint.update({
                'lagrangmdvrpn_multiplier': model.lagrangmdvrpn_multipliers,
                'lagrangmdvrpn_optimizer': model.lambda_optimizer.state_dict()
            })
        torch.save(checkpoint, save_path)

        valid_dir = os.path.join(save_dir, 'valid', str(cur_epoch))

        stc = validate(valid_loader, model, valid_dir)
        
        print('[Epoch' + str(cur_epoch) + ']' + 
              'Mean epoch reward: %2.4f, ' \
              'vailid reward: %2.4f, took: %2.4fs '% \
              (mean_reward, stc[obj], time.time() - epoch_start))
        
        if anneal:
            model.ac.tau = anneal_final + (1 - anneal_final) * (1 - cur_epoch / epochs)
        cur_epoch += 1
    # Close opened files to avoid number of open files overflow
    model.logger.close()

def main(cfg, device):
    # Load config, and start training. Do testing after training.
    ac_cfg = cfg.pop('ac_config')
    if ac_cfg is not None:
        if isinstance(ac_cfg, str):
            with open(ac_cfg, 'r') as f:
                ac_cfg = yaml.safe_load(f)

    if cfg['checkpoint'] is not None:
        print("Load checkpoint from " + cfg['checkpoint'])
        ac = load_model(cfg['checkpoint'], device)
        cfg['gnn_freeze_epochs'] = 0
    elif cfg['gnn_pretrain']:
        print("Load pretrain GNN from " + cfg['gnn_checkpoint'])
        gnn_cfg = os.path.join('/'.join(cfg['gnn_checkpoint'].split('/')[:-1]), 'config.json')
        with open(gnn_cfg, 'r') as f:
            gnn_cfg = json.load(f)
        assert gnn_cfg['gnn'].pop('embed_dim') == ac_cfg['common_cfg']['embed_dim']
        ac_cfg['encoder_cfg']['gnn_cfg']['gnn_cfg'] = gnn_cfg['gnn']
        ac = GNNAC(**ac_cfg, device=device)

        checkpoint = torch.load(cfg['gnn_checkpoint'], map_location=device)
        ac.encoder.nodes_encoder.gnn.load_state_dict(checkpoint['gnn'])     
    else:
        ac = GNNAC(**ac_cfg, device=device)
        cfg['gnn_freeze_epochs'] = 0

    RL_cfg = os.path.join("/home/fanyx/mdvrp/config", cfg['rl_algo'] + '.yaml')
    with open(RL_cfg, 'r') as f:
        RL_cfg = yaml.safe_load(f)

    save_cfg = cfg
    save_cfg.update({"ac": ac.cfg})
    save_cfg.update({"RL_cfg": RL_cfg})
    print(save_cfg)
    print(ac)

    now = time.strftime("%Y-%m-%d__%H-%M")
    save_dir = os.path.join(cfg['log_dir'], cfg['rl_algo'], now)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cfg['save_dir'] = save_dir
    with open(os.path.join(save_dir, 'config.json'), 'a') as f:
        json.dump(save_cfg, f, indent=1)

    cfg['reward_fn'] = dense_fit  
            
    if not cfg['test']:
        # Prepare Training
        train_data_list = []
        for data_set in cfg['train_data']:
            for subdir in os.listdir(data_set):
                subdir = os.path.join(data_set, subdir)
                train_data_list += [os.path.join(subdir, x) for x in os.listdir(subdir) if x.split('.')[-1] != 'txt' and x.split('.')[-1] != 'pkl']
        valid_data_list = []
        for data_set in cfg['valid_data']:
            for subdir in os.listdir(data_set):
                subdir = os.path.join(data_set, subdir)
                valid_data_list += [os.path.join(subdir, x) for x in sorted(os.listdir(subdir))[:10] if x.split('.')[-1] != 'txt' and x.split('.')[-1] != 'pkl']
        
        train_data = RLDataset(train_data_list, cfg['veh_reciprocal'], cfg.pop('cost_limits') if 'cost_limits' in cfg else None)
        valid_data = RLDataset(valid_data_list, cfg['veh_reciprocal'])
        cfg['train_data'] = train_data
        cfg['valid_data'] = valid_data
        logger_kwargs = setup_logger_kwargs(base_dir=cfg['save_dir'])
        
        model = REGISTRY[cfg['rl_algo']](
            ac=ac,
            pi_lr=cfg.pop('actor_lr'), 
            vf_lr=cfg.pop('critic_lr'),
            logger_kwargs=logger_kwargs,
            obj=cfg['obj'],
            constraint=cfg['constraint'],
            batch_size=cfg['batch_size'],
            seed=cfg.pop('seed'),
            total_epoch=cfg['epochs'],
            max_grad_norm=cfg.pop('max_grad_norm'),
            **cfg.pop('RL_cfg'),
            device=device
        )
        if cfg['checkpoint'] is not None: # Load checkpoint
            checkpoint = torch.load(cfg['checkpoint'], map_location=device)
            cfg['cur_epoch'] = 0 if cfg['recount_epoch'] else checkpoint['epoch'] + 1 
            model.epoch = cfg['cur_epoch']
            model.pi_optimizer.load_state_dict(checkpoint['actor_optim'])
            model.vf_optimizer.load_state_dict(checkpoint['critic_optim'])
            if hasattr(model, 'lagrangmdvrpn_multipliers') and 'lagrangmdvrpn_multiplier' in checkpoint:
                model.lagrangmdvrpn_multipliers = checkpoint['lagrangmdvrpn_multiplier']
                model.lambda_optimizer.load_state_dict(checkpoint['lagrangmdvrpn_optimizer'])
            # cfg['lagrangmdvrpn_multiplier'] = checkpoint['lagrangmdvrpn_multiplier']
        
        print('====================== Start Training ======================')
        train(model, **cfg)
    
    # Testing
    test_dir = os.path.join(save_dir, 'test')
    test_data_list = []
    for data_set in cfg['test_data']:
        for subdir in os.listdir(data_set):
            subdir = os.path.join(data_set, subdir)
            test_data_list += [os.path.join(subdir, x) for x in os.listdir(subdir) if x.split('.')[-1] != 'txt' and x.split('.')[-1] != 'pkl']

    test_data = RLDataset(test_data_list, cfg['veh_reciprocal'])
    test_data.use_field = True
    test_loader = RLDataLoader(dataset=test_data, batch_size=32, shuffle=False)
    stc = validate(test_loader, model, test_dir, render=True)

    print('Average distance: ' + str(stc['s']) + ', Average time: ' + str(stc['t']) + 'Average cost: ' + str(stc['c']))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatormdvrpl Optimization')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rl_algo', type=str)
    parser.add_argument('--obj', type=str, default='t')
    parser.add_argument('--constraint', nargs='*')
    parser.add_argument('--train_data', nargs='+')
    parser.add_argument('--valid_data', nargs='+')
    parser.add_argument('--test_data', nargs='*')
    parser.add_argument('--log_dir', type=str, default='runs_rl')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--final_reward', action='store_true', default=False)
    parser.add_argument('--fusing_s', action='store_true', default=False)
    parser.add_argument('--fusing_epoch', type=int, default=4)
    parser.add_argument('--total_time', action='store_true', default=False)
    parser.add_argument('--total_time_epoch', type=int, default=4)
    parser.add_argument('--veh_reciprocal', action='store_true', default=False)
    parser.add_argument('--recount_epoch', action='store_true', default=False)
    
    args = parser.parse_args()
    args = vars(args)
    dev = args.pop('device')
    if dev == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(dev if torch.cuda.is_available() else 'cpu')
    
    config = args.pop('config')
    if config is not None:
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
        config.update(args)
    else:
        config = args
    
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    #print('NOTE: SETTTING CHECKPOINT: ')
    #args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    #print(args.checkpoint)
    main(config, device)

