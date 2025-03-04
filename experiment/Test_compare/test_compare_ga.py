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
from ia.algo import MGGA
import ia.env.simulator as ia_env
from matplotlib.animation import FuncAnimation
import cv2
from tqdm import tqdm
import random

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

def random_arrangement(num_car, num_target):
    gen = {'tar': [[x, int(np.round(random.random()))] for x in range(num_target)], 'split': np.random.choice(range(num_target + 1), num_car - 1, replace = True).tolist()}
    random.shuffle(gen['tar'])
    gen['split'].sort()
    return ia_util.decode(gen)

def simulate(data_loader, model, model_GA, obj, save_dir, fig_interval = 10, pid = 0):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    r_final_ra = [[] for _ in range(6)]
    r_final = [[] for _ in range(6)]
    r_final_ga = [[] for _ in range(6)]
    tqdm_data_loader = tqdm(data_loader)
    tqdm_data_loader.set_description('Simulation')
    for batch_idx, batch in enumerate(tqdm_data_loader):
        bsz_data, info, fields, car_cfgs, _ = batch
        field = fields[0]
        car_cfg = car_cfgs[0]

        if len(car_cfg) > 1:

            T = random_arrangement(field.ori.shape[:2][0], field.ori.shape[:2][1])
            r_final_ra[field.num_fields-1].append(fit(field.D_matrix, 
                            field.ori, 
                            field.des,
                            car_cfg, 
                            field.line_length,
                            T, 
                            tS_t=False,
                            type='all'))

            with torch.no_grad():
                seq_enc, _, _, _, _, _, _ = model(bsz_data, info, deterministic=True, criticize=False)
            T = decode(seq_enc[0])
            r_final[field.num_fields-1].append(fit(field.D_matrix, 
                            field.ori, 
                            field.des,
                            car_cfg, 
                            field.line_length,
                            T, 
                            tS_t=False,
                            type='all'))

            if batch_idx % fig_interval == 0:
                simulator = arrangeSimulator(field, car_cfg)
                simulator.init_simulation(T)
                simulator.render_arrange()
                s, t, c = r_final[field.num_fields-1][-1]
                plt.title(f'$s_P$={np.round(s, 2)}m, $t_P$={np.round(t, 2)}s, $c_P$={np.round(c, 2)}L', fontsize=20)
                plt.savefig(os.path.join(save_dir, f"proc{pid}_batch{batch_idx}_mdvrp.png"))
                plt.close('all')

            T, best, log = model_GA.optimize(field.D_matrix, 
                            field.ori, 
                            car_cfg, 
                            field.line_length, 
                            field.des,
                            field.line_field_idx)

            r_final_ga[field.num_fields-1].append(fit(field.D_matrix, 
                            field.ori, 
                            field.des,
                            car_cfg, 
                            field.line_length,
                            T, 
                            tS_t=False,
                            type='all'))

            if batch_idx % fig_interval == 0:
                simulator = arrangeSimulator(field, car_cfg)
                simulator.init_simulation(T)
                simulator.render_arrange()
                s, t, c = r_final_ga[field.num_fields-1][-1]
                plt.title(f'$s_P$={np.round(s, 2)}m, $t_P$={np.round(t, 2)}s, $c_P$={np.round(c, 2)}L', fontsize=20)
                plt.savefig(os.path.join(save_dir, f"proc{pid}_batch{batch_idx}_ga.png"))
                plt.close('all')  

    r_flatten, r_flatten_ga, r_flatten_ra = [], [], []
    r_fields = [[] for _ in range(6)]
    for f_num, (r, r_ga, r_ra) in enumerate(zip(r_final, r_final_ga, r_final_ra)):
        if len(r) > 0:
            r_flatten += r
            r_flatten_ga += r_ga
            r_flatten_ra += r_ra
            mean_r = np.mean(np.array(r), axis=0)
            mean_r_ga = np.mean(np.array(r_ga), axis=0)
            mean_r_ra = np.mean(np.array(r_ra), axis=0)
            if obj == 't':
                win_num = np.sum(np.array(r)[:, 1] < np.array(r_ga)[:, 1])
            elif obj == 's': 
                win_num = np.sum(np.array(r)[:, 0] < np.array(r_ga)[:, 0])
            elif obj == 'c':
                win_num = np.sum(np.array(r)[:, 2] < np.array(r_ga)[:, 2])
            win_num = np.sum(np.array(r)[:, 1] < np.array(r_ga)[:, 1])
            print(f"Testing result of {f_num+1} field(s):")
            print(f"[RA]    | Distance:{mean_r_ra[0]:.2f} | Time: {mean_r_ra[1]:.2f} | Fuel: {mean_r_ra[2]:.2f} | Win rate: -%")            
            print(f"[mdvrp] | Distance:{mean_r[0]:.2f} | Time: {mean_r[1]:.2f} | Fuel: {mean_r[2]:.2f} | Win rate: {win_num/len(r)*100:.2f}%")
            print(f"[GA]    | Distance:{mean_r_ga[0]:.2f} | Time: {mean_r_ga[1]:.2f} | Fuel: {mean_r_ga[2]:.2f} | Win rate: {(1-win_num/len(r))*100:.2f}%")
            r_fields[f_num] = [mean_r[0], mean_r[1], mean_r[2], mean_r_ga[0], mean_r_ga[1], mean_r_ga[2], mean_r_ra[0], mean_r_ra[1], mean_r_ra[2]]

    r_final = np.array(r_flatten)
    r_final_ga = np.array(r_flatten_ga)
    r_final_ra = np.array(r_flatten_ra)
    mean_r = np.mean(r_final, axis=0)
    mean_r_ga = np.mean(r_final_ga, axis=0)
    mean_r_ra = np.mean(r_final_ra, axis=0)
    win_num = np.sum(r_final[:, 1] < r_final_ga[:, 1]) / r_final.shape[0]
    stc = {
        's': mean_r[0],
        't': mean_r[1],
        'c': mean_r[2],
        's_ga': mean_r_ga[0],
        't_ga': mean_r_ga[1],
        'c_ga': mean_r_ga[2],
        'win_rate': win_num
    }
            
    with open(os.path.join(save_dir, f'result_proc{pid}.txt'), 'w') as f:
        f.write(' '.join(map(str, r_final[:, 0].tolist())) + '\n')
        f.write(' '.join(map(str, r_final[:, 1].tolist())) + '\n')
        f.write(' '.join(map(str, r_final[:, 2].tolist())) + '\n')
        f.write(' '.join(map(str, r_final_ga[:, 0].tolist())) + '\n')
        f.write(' '.join(map(str, r_final_ga[:, 1].tolist())) + '\n')
        f.write(' '.join(map(str, r_final_ga[:, 2].tolist())) + '\n')
        f.write(' '.join(map(str, r_final_ra[:, 0].tolist())) + '\n')
        f.write(' '.join(map(str, r_final_ra[:, 1].tolist())) + '\n')
        f.write(' '.join(map(str, r_final_ra[:, 2].tolist())) + '\n')
        for sublist in r_fields:
            line = ",".join(map(str, sublist))
            f.write(line + "\n")

    
    return stc

if __name__ == '__main__':
    import argparse
    import yaml
    import time
    from data.dataset import RLDataset
    from data.dataloader import RLDataLoader
    from torch.utils.data import random_split
    import multiprocessing

    parser = argparse.ArgumentParser(description='Combinatormdvrpl Optimization')
    parser.add_argument('--checkpoint', default='/home/fanyx/mdvrp/result/training_rl/ppo/2025-02-02__08-36__t/best_model38.pt')
    parser.add_argument('--test_data', nargs='*', default=['/home/fanyx/mdvrp/data/Gdataset/Task_test_large'])
    parser.add_argument('--save_dir', type=str, default='/home/fanyx/mdvrp/experiment/Test_compare')
    parser.add_argument('--obj', type=str, default='t')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--fig_interval', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    now = time.strftime("%Y-%m-%d__%H-%M")
    save_dir = os.path.join(args.save_dir, 'GA', now)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    test_data_list = []
    for data_set in args.test_data:
        for subdir in os.listdir(data_set):
            subdir = os.path.join(data_set, subdir)
            test_data_list += [os.path.join(subdir, x) for x in os.listdir(subdir) if x.split('.')[-1] != 'txt' and x.split('.')[-1] != 'pkl']

    test_data = RLDataset(test_data_list)
    test_data.use_field = True

    total_size = len(test_data)
    print('Total size:', total_size)
    subset_size = total_size // args.num_workers
    subset_sizes = [subset_size] * args.num_workers
    subset_sizes[-1] += total_size % args.num_workers
    subsets = random_split(test_data, subset_sizes)
    dataloaders = [RLDataLoader(dataset=subset, batch_size=1, shuffle=False) for subset in subsets]

    # 定义任务函数
    def task(idx):
        ac = load_model(args.checkpoint, model_type='mdvrp')
        GA = MGGA(f = args.obj, order = True, gen_size=100, max_iter=100)
        stc = simulate(dataloaders[idx], ac, GA, args.obj, save_dir, fig_interval=args.fig_interval, pid=idx)
        print(f"Testing result of all fields:")
        print(f"[mdvrp] | Distance:{stc['s']:.2f} | Time: {stc['t']:.2f} | Fuel: {stc['c']:.2f} | Win rate: {stc['win_rate']*100:.2f}%")
        print(f"[GA]    | Distance:{stc['s_ga']:.2f} | Time: {stc['t_ga']:.2f} | Fuel: {stc['c_ga']:.2f} | Win rate: {(1-stc['win_rate'])*100:.2f}%")

    processes = []
    for i in range(args.num_workers):
        process = multiprocessing.Process(target=task, args=(i,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
