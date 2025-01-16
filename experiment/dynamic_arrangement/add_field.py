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


def simulate(data_loader, model, save_dir, model_type='mdvrp', stop_coeff=0.5, fig_interval = 10, render = False, 
             fields_num:int = 2):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    s_ori, t_ori, c_ori = [], [], []
    s_rec, t_rec, c_rec = [], [], []
    tqdm_data_loader = tqdm(data_loader)
    tqdm_data_loader.set_description('Simulation')
    for batch_idx, batch in enumerate(tqdm_data_loader):
        bsz_data, bsz_data_ia, info, fields, fields_ia, car_cfgs, _ = batch

        if model_type == 'mdvrp':
            # Full forward pass through the dataset
            with torch.no_grad():
                seq_enc, _, _, _, _, _, _ = model(bsz_data, info, deterministic=True, criticize=False)
            Ts = list(map(decode, seq_enc))
        else:
            with torch.no_grad():
                seq_enc, _, _, _, _, _, _ = model(bsz_data_ia, info, deterministic=True, criticize=False)
            Ts = list(map(ia_util.decode, seq_enc))

        for idx, (T, field, car_cfg) in enumerate(zip(Ts, fields, car_cfgs)):
            simulator = arrangeSimulator(field, car_cfg)
            simulator.init_simulation(T)

            if batch_idx % fig_interval == 0:
                ax = simulator.field.render(working_lines=False, show=False, start=True)
                t0, c0, s0, _, _, _, _ = simulator.simulate(ax, True, False, False, True)
                plt.title(f'$s_P$={np.round(s0, 2)}m, $t_P$={np.round(t0, 2)}s, $c_P$={np.round(c0, 2)}L', fontsize=20)
                plt.savefig(os.path.join(save_dir, f"batch{batch_idx}_{model_type}_ori.png"))
                plt.close('all')
            else:
                ax = None
                t0, c0, s0, _, _, _, _ = simulator.simulate(ax, False, False, False, True)
            # print(f"Original result: s={np.round(s0, 2)}m, t={np.round(t0, 2)}s, c={np.round(c0, 2)}L")
            s_ori.append(np.round(s0, 2))
            t_ori.append(np.round(t0, 2))
            c_ori.append(np.round(c0, 2))

            simulator = arrangeSimulator(field, car_cfg)
            simulator.init_simulation(T)
            simulator.simulator.time_teriminate = t0*stop_coeff
            # print(f't_term: {simulator.simulator.time_teriminate}')
            if batch_idx % fig_interval == 0:
                ax = simulator.field.render(working_lines=False, show=False, start=False)
                if render:
                    t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, True, True, False, True)
                else:
                    t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, True, False, False, True)
            else:
                ax = None
                t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, False, False, False, True)
            # print(f't_term_real: {t1}')
            if model_type == 'mdvrp':

                chosen_idx, chosen_entry = field.edit_fields(simulator.simulator.car_status)
                pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
                bsz_data['graph'] = Batch.from_data_list([pygdata])
                with torch.no_grad():
                    seq_enc, _, _, _, _, _, _ = model(bsz_data, info, deterministic=True, criticize=False, 
                                                    force_chosen=True, 
                                                    chosen_idx=chosen_idx, chosen_entry=chosen_entry)
                T = decode(seq_enc[0])
            else:
                field.edit_fields(simulator.simulator.car_status, delete_lines=False)
                chosen_idx = [[[line[0]+1 for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]]
                chosen_entry = [[[line[1] for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]]
                with torch.no_grad():
                    seq_enc, _, _, _, _, _, _ = model(bsz_data, info, deterministic=True, criticize=False, 
                                                    force_chosen=True, 
                                                    chosen_idx=chosen_idx, chosen_entry=chosen_entry)                
                T_ori = T
                T = [a[len(c):] for a, c  in zip(T_ori, chosen_idx[0])]
            
            simulator = arrangeSimulator(field, car_cfg)
            simulator.init_simulation(T)

            if batch_idx % fig_interval == 0:
                [ax.plot(field.Graph.nodes[start]['coord'][0], 
                                    field.Graph.nodes[start]['coord'][1], 
                                    '*y', 
                                    markersize=20) for start in field.starts]
                if render:
                    t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, True, True, False, False)
                    if figs1 and figs2:
                        figs = figs1 + figs2
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        videoWriter = cv2.VideoWriter(os.path.join(save_dir, f"batch{batch_idx}_{model_type}.mp4"), fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
                        for fig in figs:
                            videoWriter.write(fig)
                        videoWriter.release() 
                else:
                    t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, True, False, False, False)
                plt.title(f'$s_P$={np.round(s1+s2, 2)}m, $t_P$={np.round(t1+t2, 2)}s, $c_P$={np.round(c1+c2, 2)}L', fontsize=20)
                plt.savefig(os.path.join(save_dir, f"batch{batch_idx}_{model_type}_rec.png"))
                plt.close('all')
            else:
                t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, False, False, False, False)

            s_rec.append(np.round(s1+s2, 2))
            t_rec.append(np.round(t1+t2, 2))
            c_rec.append(np.round(c1+c2, 2))          
            
    return {'s-ori':np.mean(s_ori),
            't-ori':np.mean(t_ori),
            'c-ori':np.mean(c_ori),
            's-rec':np.mean(s_rec),
            't-rec':np.mean(t_rec),
            'c-rec':np.mean(c_rec)}

if __name__ == '__main__':
    import argparse
    import yaml
    import time
    from data.dataset import RLTestDataset
    from data.dataloader import RLTestDataLoader

    parser = argparse.ArgumentParser(description='Combinatormdvrpl Optimization')
    parser.add_argument('--checkpoint', default='/home/fanyx/mdvrp/result/training_rl/ppo/2024-12-27__12-11/best_model31.pt')
    parser.add_argument('--checkpoint_ia', default='/home/fanyx/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-04__12-53_t/best_model39.pt')
    parser.add_argument('--test_data', nargs='*', default=['/home/fanyx/mdvrp/data/Gdataset/Task_test_1d'])
    parser.add_argument('--save_dir', type=str, default='/home/fanyx/mdvrp/experiment/dynamic_arrangement')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--fig_interval', type=int, default=10)
    parser.add_argument('--stop_coeff', type=float, default=0.5)
    parser.add_argument('--render', action='store_true', default=False)
    args = parser.parse_args()
    
    # Load config, and start training. Do testing after training.
    print("Load checkpoint from " + args.checkpoint)
    ac = load_model(args.checkpoint, model_type='mdvrp')

    print("Load checkpoint from " + args.checkpoint_ia)
    ac_ia = load_model(args.checkpoint_ia, model_type='ia')

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

    test_loader = RLTestDataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    stc = simulate(test_loader, ac, save_dir, model_type='mdvrp', stop_coeff=args.stop_coeff, render=args.render, fig_interval=args.fig_interval)
    print(f"[mdvrp-ori] | Distance:{stc['s-ori']:.2f} | Time: {stc['t-ori']:.2f} | Fuel: {stc['c-ori']:.2f}")
    print(f"[mdvrp-rec] | Distance:{stc['s-rec']:.2f} | Time: {stc['t-rec']:.2f} | Fuel: {stc['c-rec']:.2f}")
    stc = simulate(test_loader, ac_ia, save_dir, model_type='ia', stop_coeff=args.stop_coeff, render=args.render, fig_interval=args.fig_interval)
    print(f"[ia-ori]    | Distance:{stc['s-ori']:.2f} | Time: {stc['t-ori']:.2f} | Fuel: {stc['c-ori']:.2f}")
    print(f"[ia-rec]    | Distance:{stc['s-rec']:.2f} | Time: {stc['t-rec']:.2f} | Fuel: {stc['c-rec']:.2f}")