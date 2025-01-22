from copy import deepcopy
from env.multi_field import multiField
from utils import fit, load_dict, decode
import os
from torch_geometric.data import Batch
import torch
from torch_geometric.utils import unbatch, from_networkx
import numpy as np
import matplotlib.pyplot as plt
from env import arrangeSimulator
import json
from model.ac import GNNAC
import ia.algo.arrangement.models.ac as ia_model
import ia.utils as ia_util
import ia.env.simulator as ia_env
from matplotlib.animation import FuncAnimation
import cv2
from tqdm import tqdm

def load_model(checkpoint, model_type='mdvrp'):
    cfg = os.path.join('/'.join(checkpoint.split('/')[:-1]), 'config.json')
    with open(cfg, 'r') as f:
        cfg = json.load(f)
    
    ac_cfg = cfg['ac']

    if model_type == 'mdvrp':
        ac = GNNAC(**ac_cfg)
    elif model_type == 'ia':
        ac = ia_model.GNNAC(**ac_cfg)
    checkpoint = torch.load(checkpoint, map_location='cpu')
    ac.load_state_dict(checkpoint['model'])  
    ac.eval()
    return ac


def simulate(data_loader, model, model_ia, save_dir, pid = 0):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    tqdm_data_loader = tqdm(data_loader)
    tqdm_data_loader.set_description('Simulation')
    for batch_idx, batch in enumerate(tqdm_data_loader):
        bsz_data, bsz_data_ia, info, fields, fields_ia, car_cfgs, _ = batch

        with torch.no_grad():
            seq_enc, _, _, _, _, _, _ = model(bsz_data, info, deterministic=True, criticize=False)
        Ts = list(map(decode, seq_enc))

        with torch.no_grad():
            seq_enc, _, _, _, _, _, _ = model_ia(bsz_data_ia, info, deterministic=True, criticize=False)
        Ts_ia = list(map(ia_util.decode, seq_enc))

        for idx, (T, T_ia, field, field_ia, car_cfg) in enumerate(zip(Ts, Ts_ia, fields, fields_ia, car_cfgs)):
            simulator = arrangeSimulator(field, car_cfg)
            simulator.init_simulation(T)
            ax = None
            t0, c0, s0, _, _, _, _ = simulator.simulate(ax, False, False, False, True)
            s, t, c = fit(field.D_matrix, 
                            field.ori.transpose(0, 1, 3, 2), 
                            field.des.transpose(0, 1, 3, 2),
                            car_cfg, 
                            field.line_length,
                            T, 
                            tS_t=False,
                            type='all')
            
            sim_ia = ia_env.arrangeSimulator(field_ia, car_cfg)
            sim_ia.init_simulation(T)
            ax = None
            t0_ia, c0, s0, _, _, _ = sim_ia.simulate(False, False, False)

            s, t_ia, c = fit(field_ia.D_matrix, 
                    np.tile(field_ia.ori, (len(car_cfg), 1, 1, 1)), 
                    np.tile(field_ia.ori, (len(car_cfg), 1, 1, 1)),
                    car_cfg, 
                    field_ia.line_length,
                    T, 
                    tS_t=False,
                    type='all')
            
            
            if abs(t0 - t) / t0 > 0.25:
                print(f't0: {t0}, t: {t}, t0_ia: {t0_ia}, t_ia: {t_ia}')
                # field.make_working_graph()
                simulator = arrangeSimulator(field, car_cfg)
                simulator.init_simulation(T)    
                ax = simulator.field.render(working_lines=False, show=False, start=True)
                t0, c0, s0, _, _, _, _ = simulator.simulate(ax, True, False, False, True)
                plt.title(f'$t_s$={np.round(t0, 2)}s, $t_f$={np.round(t, 2)}s', fontsize=20)
                plt.savefig(os.path.join(save_dir, f"proc{pid}_batch{batch_idx}_ori.png"))
                plt.close('all')   
              

if __name__ == '__main__':
    import argparse
    import yaml
    import time
    from data.dataset import RLTestDataset
    from data.dataloader import RLTestDataLoader
    from torch.utils.data import random_split
    import multiprocessing

    parser = argparse.ArgumentParser(description='Combinatormdvrpl Optimization')
    parser.add_argument('--checkpoint', default='/home/fanyx/mdvrp/result/training_rl/ppo/2024-12-27__12-11/best_model31.pt')
    parser.add_argument('--checkpoint_ia', default='/home/fanyx/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-04__12-53_t/best_model39.pt')
    parser.add_argument('--test_data', nargs='*', default=['/home/fanyx/mdvrp/data/Gdataset/Task_test_1d'])
    parser.add_argument('--save_dir', type=str, default='/home/fanyx/mdvrp/experiment/dynamic_arrangement')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--fig_interval', type=int, default=10)
    parser.add_argument('--stop_coeff', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=36)
    parser.add_argument('--render', action='store_true', default=False)
    args = parser.parse_args()

    now = time.strftime("%Y-%m-%d__%H-%M")
    save_dir = os.path.join(args.save_dir, now)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    test_data_list = []
    for data_set in args.test_data:
        for subdir in os.listdir(data_set):
            subdir = os.path.join(data_set, subdir)
            test_data_list += [os.path.join(subdir, x) for x in os.listdir(subdir) if x.split('.')[-1] != 'txt' and x.split('.')[-1] != 'pkl']

    test_data = RLTestDataset(test_data_list)
    test_data.use_field = True

    total_size = len(test_data)
    subset_size = total_size // args.num_workers
    subset_sizes = [subset_size] * args.num_workers
    subset_sizes[-1] += total_size % args.num_workers
    subsets = random_split(test_data, subset_sizes)
    dataloaders = [RLTestDataLoader(dataset=subset, batch_size=1, shuffle=False) for subset in subsets]

    # 定义任务函数
    def task(idx):
        ac = load_model(args.checkpoint, model_type='mdvrp')
        ac_ia = load_model(args.checkpoint_ia, model_type='ia')
        simulate(dataloaders[idx], ac, ac_ia, save_dir, pid=idx)

    processes = []
    for i in range(args.num_workers):
        process = multiprocessing.Process(target=task, args=(i,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
