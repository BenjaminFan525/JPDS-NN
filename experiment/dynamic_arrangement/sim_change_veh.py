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


def simulate(data_loader, model, model_ia, save_dir, stop_coeff=0.5, fig_interval = 10, pid = 0, render = False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    s_ori, t_ori, c_ori = [], [], []
    s_rec, t_rec, c_rec = [], [], []
    s_rec_ia, t_rec_ia, c_rec_ia = [], [], []
    tqdm_data_loader = tqdm(data_loader)
    tqdm_data_loader.set_description('Simulation')
    for batch_idx, batch in enumerate(tqdm_data_loader):
        b = deepcopy(batch)
        bsz_data, bsz_data_ia, info, fields, fields_ia, car_cfgs, _ = b
        for idx, (field, field_ia, car_cfg) in enumerate(zip(fields, fields_ia, car_cfgs)):         
            if len(car_cfg) == 1:
                veh_del = []
            elif len(car_cfg) <= 3:
                veh_del = random.sample(range(len(car_cfg)), len(car_cfg)-1)
            else:
                veh_del = random.sample(range(len(car_cfg)), len(car_cfg)-2)
            num_veh = len(car_cfg) - len(veh_del)                

            with torch.no_grad():
                seq_enc, _, _, _, _, _, _ = model(bsz_data, info, deterministic=True, criticize=False, )
            T = decode(seq_enc[0])

            simulator = arrangeSimulator(field, car_cfg)
            simulator.init_simulation(T, debug=True)
            t, c, s, _, _, _, _ = simulator.simulate(None, False, False, False, False)

            with torch.no_grad():
                seq_enc, _, _, _, _, _, _ = model_ia(bsz_data_ia, info, deterministic=True, criticize=False, )
            T_ia = ia_util.decode(seq_enc[0])            
            
            simulator = arrangeSimulator(field, car_cfg)
            simulator.init_simulation(T_ia, debug=True)
            t_ia, c_ia, s_ia, _, _, _, _ = simulator.simulate(None, False, False, False, False)            
            
            if t < t_ia:
                T0 = T
                t0 = t
            else:
                T0 = T_ia
                t0 = t_ia
          
            simulator = arrangeSimulator(field, car_cfg)
            simulator.init_simulation(T0)

            if batch_idx % fig_interval == 0:
                ax = simulator.field.render(working_lines=False, show=False, start=True)
                t0, c0, s0, _, _, _, _ = simulator.simulate(ax, True, False, False, True)
                plt.title(f'$s_P$={np.round(s0, 2)}m, $t_P$={np.round(t0, 2)}s, $c_P$={np.round(c0, 2)}L', fontsize=20)
                plt.savefig(os.path.join(save_dir, f"proc{pid}_batch{batch_idx}_ori.png"))
                plt.close('all')
            else:
                ax = None
                t0, c0, s0, _, _, _, _ = simulator.simulate(ax, False, False, False, True)
            # print(f"Original result: s={np.round(s0, 2)}m, t={np.round(t0, 2)}s, c={np.round(c0, 2)}L")
            s_ori.append(np.round(s0, 2))
            t_ori.append(np.round(t0, 2))
            c_ori.append(np.round(c0, 2))

        b = deepcopy(batch)
        bsz_data, bsz_data_ia, info, fields, fields_ia, car_cfgs, _ = b
        for idx, (field, field_ia, car_cfg) in enumerate(zip(fields, fields_ia, car_cfgs)):
            simulator = arrangeSimulator(field, car_cfg)
            simulator.init_simulation(T0)
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

            car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for idx, cur_car in enumerate(car_cfg) if idx not in veh_del])).float()
            chosen_idx_full, chosen_entry_full = field.edit_fields(simulator.simulator.car_status, veh_del=veh_del)
            T_full, chosen_idx, chosen_entry = [], [], []
            free_line, free_entry = [], []
            veh_key_padding_mask = torch.zeros((1, car_tensor.shape[0])).bool()
            for idx, (line, ent) in enumerate(zip(chosen_idx_full[0], chosen_entry_full[0])):
                if idx in veh_del:
                    if len(line) and len(ent):
                        T_full.append([[line[0] - 2*num_veh, ent[0]]])
                        free_line.append(line[0])
                        free_entry.append(ent[0])
                    else:
                        T_full.append([])
                else:
                    T_full.append([])
                    if len(line) and len(ent):
                        chosen_idx.append([line[0]])
                        chosen_entry.append([ent[0]])
                    else:
                        chosen_idx.append([])
                        chosen_entry.append([])                        
            chosen_idx.append(free_line)
            chosen_entry.append(free_entry)
            chosen_idx = [chosen_idx]
            chosen_entry = [chosen_entry]
            pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
            bsz_data['graph'] = Batch.from_data_list([pygdata])
            bsz_data['vehicles'] = car_tensor.unsqueeze(0)
            info = {'veh_key_padding_mask': veh_key_padding_mask, 'num_veh': torch.tensor([[num_veh]])}
            with torch.no_grad():
                seq_enc, _, _, _, _, _, _ = model(bsz_data, info, deterministic=True, criticize=False, 
                                                force_chosen=True, 
                                                chosen_idx=chosen_idx, chosen_entry=chosen_entry)
            T = decode(seq_enc[0])

            t_idx = 0
            for idx, a in enumerate(T_full):
                if idx not in veh_del:
                    T_full[idx] = T[t_idx]
                    t_idx += 1
            field.starts = [f'start-{idx}' for idx in range(len(car_cfg))]
            field.ends = [field.ends[0]]*len(car_cfg)
            field.num_veh = len(field.starts)
            field.num_endpoints = 2*field.num_veh
            field.make_working_graph()
            
            simulator = arrangeSimulator(field, car_cfg)
            simulator.init_simulation(T_full)

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
                        videoWriter = cv2.VideoWriter(os.path.join(save_dir, f"batch{batch_idx}_mdvrp.mp4"), fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
                        for fig in figs:
                            videoWriter.write(fig)
                        videoWriter.release() 
                else:
                    t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, True, False, False, False)
                plt.title(f'$s_P$={np.round(s1+s2, 2)}m, $t_P$={np.round(t1+t2, 2)}s, $c_P$={np.round(c1+c2, 2)}L', fontsize=20)
                plt.savefig(os.path.join(save_dir, f"proc{pid}_batch{batch_idx}_mdvrp_rec.png"))
                plt.close('all')
            else:
                ax = None
                t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, False, False, False, False)            
            
            s_rec.append(np.round(s1+s2, 2))
            t_rec.append(np.round(t1+t2, 2))
            c_rec.append(np.round(c1+c2, 2))  
            
        b = deepcopy(batch)
        bsz_data, bsz_data_ia, info, fields, fields_ia, car_cfgs, _ = b
        for idx, (field, field_ia,car_cfg) in enumerate(zip(fields, fields_ia, car_cfgs)):
            simulator = arrangeSimulator(field, car_cfg)
            simulator.init_simulation(T0)
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

            car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for idx, cur_car in enumerate(car_cfg) if idx not in veh_del])).float()
            chosen_line = [[field_ia.working_line_list[line[0]] for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]
            field.edit_fields(simulator.simulator.car_status, delete_lines=False)
            chosen_idx_full = [[[list(field_ia.working_graph.nodes()).index(line) for line in chosen_line[idx]] for idx in range(len(car_cfg))]]
            chosen_entry_full = [[[line[1] for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]]

            for idx, status in enumerate(simulator.simulator.car_status):
                if status['inline']:
                    chosen_idx_full[0][idx].append(list(field_ia.working_graph.nodes()).index(status['line']))
                    chosen_entry_full[0][idx].append(status['entry'])

            T_full = [[] for _ in range(len(car_cfg))]
            chosen_idx, chosen_entry = [], []
            free_line, free_entry = [], []
            veh_key_padding_mask = torch.zeros((1, len(car_tensor))).bool()
            for idx, (line, ent, status) in enumerate(zip(chosen_idx_full[0], chosen_entry_full[0], simulator.simulator.car_status)):
                if idx in veh_del:
                    if status['inline']:
                        if len(line) and len(ent):
                            T_full[idx].append([line[-1] - 1, ent[-1]])
                    free_line += line
                    free_entry += ent
                else:
                    chosen_idx.append(line)
                    chosen_entry.append(ent)
            chosen_idx.append(free_line)
            chosen_entry.append(free_entry)
            chosen_idx = [chosen_idx]
            chosen_entry = [chosen_entry]

            info = {'veh_key_padding_mask': veh_key_padding_mask, 'num_veh': torch.tensor([[num_veh]])}
            bsz_data_ia['vehicles'] = car_tensor.unsqueeze(0)
            with torch.no_grad():
                seq_enc, _, _, _, _, _, _ = model_ia(bsz_data_ia, info, deterministic=True, criticize=False, 
                                                force_chosen=True, 
                                                chosen_idx=chosen_idx, chosen_entry=chosen_entry)     
                           
            T_ia_2 = ia_util.decode(seq_enc[0])
            t_idx = 0
            for idx, (a, c, status) in enumerate(zip(T_full, chosen_idx_full[0], simulator.simulator.car_status)):
                if idx not in veh_del:
                    if status['inline']:
                        T_full[idx] = T_ia_2[t_idx][len(c)-1:]
                    else:
                        T_full[idx] = T_ia_2[t_idx][len(c):]
                    t_idx += 1

            field.starts = [f'start-{idx}' for idx in range(len(car_cfg))]
            field.ends = [field.ends[0]]*len(car_cfg)
            field.num_veh = len(field.starts)
            field.num_endpoints = 2*field.num_veh
            field.make_working_graph()
            
            simulator = arrangeSimulator(field, car_cfg)
            simulator.init_simulation(T_full)

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
                        videoWriter = cv2.VideoWriter(os.path.join(save_dir, f"batch{batch_idx}_ia.mp4"), fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
                        for fig in figs:
                            videoWriter.write(fig)
                        videoWriter.release() 
                else:
                    t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, True, False, False, False)
                plt.title(f'$s_P$={np.round(s1+s2, 2)}m, $t_P$={np.round(t1+t2, 2)}s, $c_P$={np.round(c1+c2, 2)}L', fontsize=20)
                plt.savefig(os.path.join(save_dir, f"proc{pid}_batch{batch_idx}_ia_rec.png"))
                plt.close('all')
            else:
                ax = None
                t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, False, False, False, False)

            s_rec_ia.append(np.round(s1+s2, 2))
            t_rec_ia.append(np.round(t1+t2, 2))
            c_rec_ia.append(np.round(c1+c2, 2))          
            
    with open(os.path.join(save_dir, f'result_proc{pid}_mdvrp.txt'), 'w') as f:
        for lst in [s_ori, t_ori, c_ori, s_rec, t_rec, c_rec]:
            f.write(' '.join(map(str, lst)) + '\n')

    with open(os.path.join(save_dir, f'result_proc{pid}_ia.txt'), 'w') as f:
        for lst in [s_ori, t_ori, c_ori, s_rec_ia, t_rec_ia, c_rec_ia]:
            f.write(' '.join(map(str, lst)) + '\n')

    return {'s-ori':np.mean(s_ori),
            't-ori':np.mean(t_ori),
            'c-ori':np.mean(c_ori),
            's-rec':np.mean(s_rec),
            't-rec':np.mean(t_rec),
            'c-rec':np.mean(c_rec),
            's-rec-ia':np.mean(s_rec_ia),
            't-rec-ia':np.mean(t_rec_ia),
            'c-rec-ia':np.mean(c_rec_ia)}

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
    parser.add_argument('--test_data', nargs='*', default=['/home/fanyx/mdvrp/data/Gdataset/Task_test_debug'])
    parser.add_argument('--save_dir', type=str, default='/home/fanyx/mdvrp/experiment/dynamic_arrangement')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--fig_interval', type=int, default=10)
    parser.add_argument('--stop_coeff', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=1)
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
        stc = simulate(dataloaders[idx], ac, ac_ia, save_dir, stop_coeff=args.stop_coeff, render=args.render, fig_interval=args.fig_interval, pid=idx)
        print(f"[mdvrp-ori] | Distance:{stc['s-ori']:.2f} | Time: {stc['t-ori']:.2f} | Fuel: {stc['c-ori']:.2f}")
        print(f"[mdvrp-rec] | Distance:{stc['s-rec']:.2f} | Time: {stc['t-rec']:.2f} | Fuel: {stc['c-rec']:.2f}")
        print(f"[ia-ori]    | Distance:{stc['s-ori']:.2f} | Time: {stc['t-ori']:.2f} | Fuel: {stc['c-ori']:.2f}")
        print(f"[ia-rec]    | Distance:{stc['s-rec-ia']:.2f} | Time: {stc['t-rec-ia']:.2f} | Fuel: {stc['c-rec-ia']:.2f}")

    processes = []
    for i in range(args.num_workers):
        process = multiprocessing.Process(target=task, args=(i,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
