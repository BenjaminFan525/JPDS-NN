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
from algo import MGGA
import ia.algo.arrangement.models.ac as ia_model
import ia.utils as ia_util
import ia.env.simulator as ia_env
from matplotlib.animation import FuncAnimation
import cv2
import copy

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
# data = "/home/fanyx/mdvrp/data/Gdataset/Task_test_debug/6_4/3_28_24"
data = "/home/fanyx/mdvrp/data/Gdataset/Task_test_multifield/single_depot/4_4/10_88_12"

field = load_dict(os.path.join(data, 'field.pkl'))

car_cfg = load_dict(os.path.join(data, 'car_cfg.pkl'))
# car_cfg = [{'vw': 2.5, 'vv': 4.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
#             {'vw': 3, 'vv': 5, 'cw': 0.007, 'cv': 0.005, 'tt': 0, 'min_R': 6},
#             {'vw': 3, 'vv': 5.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
#             {'vw': 2, 'vv': 5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
#             {'vw': 3, 'vv': 6, 'cw': 0.01, 'cv': 0.008, 'tt': 0, 'min_R': 6},]

algo = 'PPO-t'
f = 't'

checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2025-02-02__08-36__t/best_model38.pt"

car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for cur_car in car_cfg])).float()

# ============================ MDVRP model ============================

ac = load_model(checkpoint)
ac.sequential_sel = True

pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
veh_key_padding_mask = torch.zeros((1, len(car_tensor))).bool()
info = {'veh_key_padding_mask': veh_key_padding_mask, 'num_veh': torch.tensor([[car_tensor.shape[0]]])}

with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, )
T0 = decode(seq_enc[0])

s, t, c = fit(field.D_matrix, 
        field.ori, 
        field.des,
        car_cfg, 
        field.line_length,
        T0, 
        tS_t=False,
        type='all')
print(f"Fit result: s={np.round(s, 2)}m, t={np.round(t, 2)}s, c={np.round(c, 2)}L")
# ============================ dynamic arrangement ============================
print(f"Simulator initializng...")
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T0, debug=True)
simulator.simulator.time_teriminate = t*0.5
ax = simulator.field.render(working_lines=False, show=False, start=False)
t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, True, True, True, True)
print(f"Simulator paused")
for idx, status in enumerate(simulator.simulator.car_status):
    print(f"Car {idx+1} | line {status['line']} | entry {status['entry']} | pos ({status['pos'][0]:3.2f}, {status['pos'][1]:3.2f}) | inline {status['inline']}")

veh_del = [0, 2]
car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for idx, cur_car in enumerate(car_cfg) if idx not in veh_del])).float()

num_veh = car_tensor.shape[0]
chosen_idx_full, chosen_entry_full = field.edit_fields(simulator.simulator.car_status, veh_del=veh_del)
T_full, chosen_idx, chosen_entry = [], [], []
free_line, free_entry = [], []
veh_key_padding_mask = torch.zeros((1, len(car_tensor))).bool()
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
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
info = {'veh_key_padding_mask': veh_key_padding_mask, 'num_veh': torch.tensor([[num_veh]])}

with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, 
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

print(f"Simulator restarting...")
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T_full)
[ax.plot(field.Graph.nodes[start]['coord'][0], 
                    field.Graph.nodes[start]['coord'][1], 
                    '*y', 
                    markersize=20) for start in field.starts]
t2, c2, s2, _, _, _, figs2 = simulator.simulate(ax, True, True, True, False)

print(f"Simulation result: s={np.round(s1+s2, 2)}m, t={np.round(t1+t2, 2)}s, c={np.round(c1+c2, 2)}L")

figs = figs1 + figs2
save_dir = os.path.join(data, algo + "_veh.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(save_dir, fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
# map(videoWriter.write, figs)
for fig in figs:
    videoWriter.write(fig)
videoWriter.release() 
    
print(save_dir)
plt.close('all')

# ============================ dynamic arrangement ============================
field = load_dict(os.path.join(data, 'field.pkl'))
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T0, debug=True)
simulator.simulator.time_teriminate = t*0.5
ax = simulator.field.render(working_lines=False, show=False, start=False)
t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, True, True, True, True)

veh_del = [0, 2]
car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for idx, cur_car in enumerate(car_cfg) if idx not in veh_del])).float()
car_cfg_del = [cur_car for idx, cur_car in enumerate(car_cfg) if idx not in veh_del]
num_veh = car_tensor.shape[0]
chosen_idx_full, chosen_entry_full = field.edit_fields(simulator.simulator.car_status, veh_del=veh_del)
T_full, chosen_idx, chosen_entry = [], [], []
free_line, free_entry = [], []
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

GA = MGGA(f = f, order = True, gen_size=100, max_iter=100)
T, best, log = GA.optimize(field.D_matrix, 
                field.ori, 
                car_cfg_del, 
                field.line_length, 
                field.des,
                field.line_field_idx,
                force_chosen=True,chosen_entry=chosen_entry[0], chosen_idx=chosen_idx[0])

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

print(f"Simulator restarting...")
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T_full)
[ax.plot(field.Graph.nodes[start]['coord'][0], 
                    field.Graph.nodes[start]['coord'][1], 
                    '*y', 
                    markersize=20) for start in field.starts]
t2, c2, s2, _, _, _, figs2 = simulator.simulate(ax, True, True, True, False)

print(f"Simulation result: s={np.round(s1+s2, 2)}m, t={np.round(t1+t2, 2)}s, c={np.round(c1+c2, 2)}L")

figs = figs1 + figs2
algo = 'GA'
save_dir = os.path.join(data, algo + "_veh.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(save_dir, fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
# map(videoWriter.write, figs)
for fig in figs:
    videoWriter.write(fig)
videoWriter.release() 
    
print(save_dir)