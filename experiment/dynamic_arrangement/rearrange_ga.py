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
from algo import MGGA
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


def simulate(data_loader, model, model_GA, save_dir, stop_coeff=0.5, fig_interval = 10, pid = 0):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    r_final = [[] for _ in range(3)]
    r_final_ga = [[] for _ in range(3)]
    tqdm_data_loader = tqdm(data_loader)
    tqdm_data_loader.set_description('Simulation')
    for batch_idx, batch in enumerate(tqdm_data_loader):
        bsz_data, info, fields, car_cfgs, _ = deepcopy(batch)
        field = fields[0]
        car_cfg = car_cfgs[0]

        with torch.no_grad():
            seq_enc, _, _, _, _, _, _ = model(bsz_data, info, deterministic=True, criticize=False)
        T0 = decode(seq_enc[0])
        s, t, c = fit(field.D_matrix, 
                        field.ori, 
                        field.des,
                        car_cfg, 
                        field.line_length,
                        T0, 
                        tS_t=False,
                        type='all')

        simulator = arrangeSimulator(field, car_cfg)
        simulator.init_simulation(T0)
        simulator.simulator.time_teriminate = t*stop_coeff
        if batch_idx % fig_interval == 0:
            ax = simulator.field.render(working_lines=False, show=False, start=False)
            t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, True, False, False, True)
        else:
            ax = None
            t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, False, False, False, True)
            
        chosen_idx, chosen_entry = field.edit_fields(simulator.simulator.car_status)
        if len(field.line_length) <= len(car_cfg):
            continue
        pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
        bsz_data['graph'] = Batch.from_data_list([pygdata])
        with torch.no_grad():
            seq_enc, _, _, _, _, _, _ = model(bsz_data, info, deterministic=True, criticize=False, 
                                            force_chosen=True, 
                                            chosen_idx=chosen_idx, chosen_entry=chosen_entry)
        T = decode(seq_enc[0])
            
        simulator = arrangeSimulator(field, car_cfg)
        simulator.init_simulation(T)

        if batch_idx % fig_interval == 0:
            [ax.plot(field.Graph.nodes[start]['coord'][0], 
                                field.Graph.nodes[start]['coord'][1], 
                                '*y', 
                                markersize=20) for start in field.starts]
            t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, True, False, False, False)
            plt.title(f'$s_P$={np.round(s1+s2, 2)}m, $t_P$={np.round(t1+t2, 2)}s, $c_P$={np.round(c1+c2, 2)}L', fontsize=20)
            plt.savefig(os.path.join(save_dir, f"proc{pid}_batch{batch_idx}_mdvrp_rec.png"))
            plt.close('all')
        else:
            ax = None
            t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, False, False, False, False)

        r_final[0].append(np.round(s1+s2, 2))
        r_final[1].append(np.round(t1+t2, 2))
        r_final[2].append(np.round(c1+c2, 2))   

        bsz_data, info, fields, car_cfgs, _ = deepcopy(batch)
        field = fields[0]
        car_cfg = car_cfgs[0]    

        simulator = arrangeSimulator(field, car_cfg)
        simulator.init_simulation(T0, debug=True)
        simulator.simulator.time_teriminate = t*stop_coeff
        if batch_idx % fig_interval == 0:
            ax = simulator.field.render(working_lines=False, show=False, start=False)
            t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, True, False, False, True)
        else:
            ax = None
            t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, False, False, False, True)
        chosen_idx, chosen_entry = field.edit_fields(simulator.simulator.car_status)
        
        GA = MGGA(f = 't', order = True, gen_size=100, max_iter=100)
        T, best, log = GA.optimize(field.D_matrix, 
                        field.ori, 
                        car_cfg, 
                        field.line_length, 
                        field.des,
                        field.line_field_idx,
                        force_chosen=True,chosen_entry=chosen_entry[0], chosen_idx=chosen_idx[0])
        simulator = arrangeSimulator(field, car_cfg)
        simulator.init_simulation(T)

        if batch_idx % fig_interval == 0:
            [ax.plot(field.Graph.nodes[start]['coord'][0], 
                                field.Graph.nodes[start]['coord'][1], 
                                '*y', 
                                markersize=20) for start in field.starts]
            t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, True, False, False, False)
            plt.title(f'$s_P$={np.round(s1+s2, 2)}m, $t_P$={np.round(t1+t2, 2)}s, $c_P$={np.round(c1+c2, 2)}L', fontsize=20)
            plt.savefig(os.path.join(save_dir, f"proc{pid}_batch{batch_idx}_ga_rec.png"))
            plt.close('all')
        else:
            ax = None
            t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, False, False, False, False)

        r_final_ga[0].append(np.round(s1+s2, 2))
        r_final_ga[1].append(np.round(t1+t2, 2))
        r_final_ga[2].append(np.round(c1+c2, 2))
            
    r_final = np.array(r_final)
    r_final_ga = np.array(r_final_ga)
    mean_r = np.mean(r_final, axis=1)
    mean_r_ga = np.mean(r_final_ga, axis=1)
    win_num = np.sum(r_final[1, :] < r_final_ga[1, :]) / r_final.shape[0]
    stc = {
        's': mean_r[0],
        't': mean_r[1],
        'c': mean_r[2],
        's_ga': mean_r_ga[0],
        't_ga': mean_r_ga[1],
        'c_ga': mean_r_ga[2],
    }
            
    with open(os.path.join(save_dir, f'result_proc{pid}.txt'), 'w') as f:
        f.write(' '.join(map(str, r_final[0, :].tolist())) + '\n')
        f.write(' '.join(map(str, r_final[1, :].tolist())) + '\n')
        f.write(' '.join(map(str, r_final[2, :].tolist())) + '\n')
        f.write(' '.join(map(str, r_final_ga[0, :].tolist())) + '\n')
        f.write(' '.join(map(str, r_final_ga[1, :].tolist())) + '\n')
        f.write(' '.join(map(str, r_final_ga[2, :].tolist())) + '\n')
    
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
    parser.add_argument('--test_data', nargs='*', default=['/home/fanyx/mdvrp/data/Gdataset/Task_test_debug'])
    parser.add_argument('--save_dir', type=str, default='/home/fanyx/mdvrp/experiment/dynamic_arrangement')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--fig_interval', type=int, default=1)
    parser.add_argument('--stop_coeff', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=50)
    args = parser.parse_args()

    now = time.strftime("%Y-%m-%d__%H-%M")
    save_dir = os.path.join(args.save_dir, now+'_rearrange_ga')
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
    subset_size = total_size // args.num_workers
    subset_sizes = [subset_size] * args.num_workers
    subset_sizes[-1] += total_size % args.num_workers
    subsets = random_split(test_data, subset_sizes)
    dataloaders = [RLDataLoader(dataset=subset, batch_size=1, shuffle=False) for subset in subsets]

    # 定义任务函数
    def task(idx):
        ac = load_model(args.checkpoint, model_type='mdvrp')
        GA = MGGA(f = 't', order = True, gen_size=100, max_iter=100)
        stc = simulate(dataloaders[idx], ac, GA, save_dir, stop_coeff=args.stop_coeff, fig_interval=args.fig_interval, pid=idx)
        print(f"[mdvrp] | Distance:{stc['s']:.2f} | Time: {stc['t']:.2f} | Fuel: {stc['c']:.2f}")
        print(f"[GA]    | Distance:{stc['s_ga']:.2f} | Time: {stc['t_ga']:.2f} | Fuel: {stc['c_ga']:.2f}")

    processes = []
    for i in range(args.num_workers):
        process = multiprocessing.Process(target=task, args=(i,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()