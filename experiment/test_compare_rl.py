import os
import time
import argparse
import numpy as np
import torch
from model.ac import GNNAC
import ia.algo.arrangement.models.ac as ia_model
import ia.utils as ia_util
from data.dataset import RLTestDataset
from data.dataloader import RLTestDataLoader
import yaml
from utils import dense_fit, fit, decode
from algo.buffer import Buffer
import matplotlib.pyplot as plt
import json
from algo import REGISTRY
from utils.logger import setup_logger_kwargs
from algo import PG
from env import arrangeSimulator
import ia.env.simulator as ia_env
from copy import deepcopy
from tqdm import tqdm


def load_model(checkpoint, device, model_type='mdvrp'):
    cfg = os.path.join('/'.join(checkpoint.split('/')[:-1]), 'config.json')
    with open(cfg, 'r') as f:
        cfg = json.load(f)
    
    ac_cfg = cfg['ac']

    if model_type == 'mdvrp':
        ac = GNNAC(**ac_cfg, device=device)
    elif model_type == 'ia':
        ac = ia_model.GNNAC(**ac_cfg, device=device)
    checkpoint = torch.load(checkpoint, map_location=device)
    ac.load_state_dict(checkpoint['model'])  
    return ac


def validate(data_loader, model, model_ia, save_dir='.', render=False):
    """Used to monitor progress on a validation set & optionally plot solution."""
    # tic = time.time()
    model.eval()
    model_ia.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if render:
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    r_final = [[] for _ in range(6)]
    r_final_ia = [[] for _ in range(6)]
    tqdm_data_loader = tqdm(data_loader)
    tqdm_data_loader.set_description('Validation')
    for batch_idx, batch in enumerate(tqdm_data_loader):

        if render:
            bsz_data, bsz_data_ia, info, fields, fields_ia, car_cfgs, _ = batch
            field_matrices = [[field.D_matrix, field.ori, field.des, field.line_length, field.num_fields] for field in fields]
            field_matrices_ia = [[field.D_matrix, field.ori, field.ori, field.line_length, field.num_fields] for field in fields_ia]
        else:
            bsz_data, bsz_data_ia, info, field_matrices, field_matrices_ia, car_cfgs, _ = batch


        for key, _ in bsz_data.items():
            bsz_data[key] = bsz_data[key].to(device)
        for key, _ in bsz_data_ia.items():
            bsz_data_ia[key] = bsz_data_ia[key].to(device)
        for key, _ in info.items():
            info[key] = info[key].to(device)
        
        # Full forward pass through the dataset
        with torch.no_grad():
            seq_enc, _, _, _, _, _, _ = model(bsz_data, info, deterministic=True, criticize=False)

        with torch.no_grad():
            seq_enc_ia, _, _, _, _, _, _ = model_ia(bsz_data_ia, info, deterministic=True, criticize=False)

        Ts = list(map(decode, seq_enc)) # decode the arrangements
        Ts_ia = list(map(ia_util.decode, seq_enc_ia))

        for bidx, (T, T_ia, field_matrix, field_matrix_ia, car_cfg) in enumerate(zip(Ts, Ts_ia, field_matrices, field_matrices_ia, car_cfgs)):
            r_final[field_matrix[-1]-1].append(fit(field_matrix[0], 
                    field_matrix[1], 
                    field_matrix[2],
                    car_cfg, 
                    field_matrix[3],
                    T, 
                    tS_t=False,
                    type='all'))
            
            r_final_ia[field_matrix_ia[-1]-1].append(fit(field_matrix_ia[0], 
                    np.tile(field_matrix_ia[1], (len(car_cfg), 1, 1, 1)),
                    np.tile(field_matrix_ia[2], (len(car_cfg), 1, 1, 1)),
                    car_cfg, 
                    field_matrix_ia[3],
                    T_ia, 
                    tS_t=False,
                    type='all'))
            if bidx == 0:
                s, t, c = r_final[field_matrix[-1]-1][-1]
                s_ia, t_ia, c_ia = r_final_ia[field_matrix[-1]-1][-1]
                
        if render:
            name = 'batch%d.png'%(batch_idx)
            path = os.path.join(save_dir, name)
            sim = arrangeSimulator(fields[0], car_cfgs[0])
            sim.init_simulation(Ts[0], init_simulator=False)
            sim.render_arrange(ax1)
            ax1.set_title(f'$s_P$={np.round(s, 2)}m, $t_P$={np.round(t, 2)}s, $c_P$={np.round(c, 2)}L', fontsize=20)
            ax1.axis('equal')
            sim_ia = ia_env.arrangeSimulator(fields_ia[0], car_cfgs[0])
            sim_ia.init_simulation(Ts_ia[0], init_simulator=False)
            sim_ia.render_arrange(ax2)
            ax2.set_title(f'$s_P$={np.round(s_ia, 2)}m, $t_P$={np.round(t_ia, 2)}s, $c_P$={np.round(c_ia, 2)}L', fontsize=20)
            ax2.axis('equal')
            plt.tight_layout()
            plt.savefig(path)
            ax1.clear()
            ax2.clear()

    plt.close('all')
    model.train()
    model_ia.train()

    r_flatten, r_flatten_ia = [], []
    for f_num, (r, r_ia) in enumerate(zip(r_final, r_final_ia)):
        if len(r) > 0:
            r_flatten += r
            r_flatten_ia += r_ia
            mean_r = np.mean(np.array(r), axis=0)
            mean_r_ia = np.mean(np.array(r_ia), axis=0)
            win_num = np.sum(np.array(r)[:, 1] < np.array(r_ia)[:, 1])
            print(f"Testing result of {f_num+1} field(s):")
            print(f"[mdvrp] | Distance:{mean_r[0]:.2f} | Time: {mean_r[1]:.2f} | Fuel: {mean_r[2]:.2f} | Win rate: {win_num/len(r)*100:.2f}%")
            print(f"[ia]    | Distance:{mean_r_ia[0]:.2f} | Time: {mean_r_ia[1]:.2f} | Fuel: {mean_r_ia[2]:.2f} | Win rate: {(1-win_num/len(r))*100:.2f}%")

    r_final = np.array(r_flatten)
    r_final_ia = np.array(r_flatten_ia)
    mean_r = np.mean(r_final, axis=0)
    mean_r_ia = np.mean(r_final_ia, axis=0)
    win_num = np.sum(r_final[:, 1] < r_final_ia[:, 1]) / r_final.shape[0]
    stc = {
        's': mean_r[0],
        't': mean_r[1],
        'c': mean_r[2],
        's_ia': mean_r_ia[0],
        't_ia': mean_r_ia[1],
        'c_ia': mean_r_ia[2],
        'win_rate': win_num
    }
    # print('Validating time: ' + str(time.time() - tic))
    return stc


def main(cfg, device):
    # Load config, and start training. Do testing after training.
    ac_cfg = cfg.pop('ac_config')
    if ac_cfg is not None:
        if isinstance(ac_cfg, str):
            with open(ac_cfg, 'r') as f:
                ac_cfg = yaml.safe_load(f)

    print("Load checkpoint from " + cfg['checkpoint'])
    ac = load_model(cfg['checkpoint'], device, model_type='mdvrp')

    print("Load checkpoint from " + cfg['checkpoint_ia'])
    ac_ia = load_model(cfg['checkpoint_ia'], device, model_type='ia')

    RL_cfg = os.path.join("/home/fanyx/mdvrp/config", cfg['rl_algo'] + '.yaml')
    with open(RL_cfg, 'r') as f:
        RL_cfg = yaml.safe_load(f)

    save_cfg = cfg
    save_cfg.update({"ac": ac.cfg})
    save_cfg.update({"RL_cfg": RL_cfg})

    now = time.strftime("%Y-%m-%d__%H-%M")
    save_dir = os.path.join(cfg['log_dir'], cfg['rl_algo'], now)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cfg['save_dir'] = save_dir
    
    # Testing
    test_data_list = []
    for data_set in cfg['test_data']:
        for subdir in os.listdir(data_set):
            subdir = os.path.join(data_set, subdir)
            test_data_list += [os.path.join(subdir, x) for x in os.listdir(subdir) if x.split('.')[-1] != 'txt' and x.split('.')[-1] != 'pkl']

    test_data = RLTestDataset(test_data_list, cfg['veh_reciprocal'])
    test_data.use_field = True
    test_loader = RLTestDataLoader(dataset=test_data, batch_size=cfg['batch_size'], shuffle=False)
    stc = validate(test_loader, ac, ac_ia, save_dir, render=True)
    print(f"Testing result of all fields:")
    print(f"[mdvrp] | Distance:{stc['s']:.2f} | Time: {stc['t']:.2f} | Fuel: {stc['c']:.2f} | Win rate: {stc['win_rate']*100:.2f}%")
    print(f"[ia]    | Distance:{stc['s_ia']:.2f} | Time: {stc['t_ia']:.2f} | Fuel: {stc['c_ia']:.2f} | Win rate: {(1-stc['win_rate'])*100:.2f}%")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatormdvrpl Optimization')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint_ia', default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rl_algo', type=str)
    parser.add_argument('--obj', type=str, default='t')
    parser.add_argument('--test_data', nargs='*')
    parser.add_argument('--log_dir', type=str, default='runs_rl')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--veh_reciprocal', action='store_true', default=False)
    
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

