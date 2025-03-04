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
import random
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

data = "/home/fanyx/mdvrp/data/Gdataset/Task_test_multifield/single_depot/4_4/60_77_12"

field = load_dict(os.path.join(data, 'field.pkl'))
field.starts = ['bound-3-2' for _ in range(4)]
field.ends = ['bound-3-2' for _ in range(4)]
field.make_working_graph()

car_cfg = load_dict(os.path.join(data, 'car_cfg.pkl'))

algo = 'PPO'
# f = 't'
# checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2025-02-02__08-36__t/best_model38.pt"
# f = 's'
# checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2025-01-31__10-37__s/best_model19.pt"
f = 'c'
checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2025-01-31__10-38__c/best_model22.pt"

car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for cur_car in car_cfg])).float()

# ============================ RA ============================
T0 = [[[22, 0], [40, 0], [41, 0], [11, 0], [0, 1], [19, 0]], 
      [[17, 1], [5, 1], [28, 1], [58, 0], [25, 1], [21, 0], [7, 0], [62, 0], [24, 1], [57, 0], [44, 0], [3, 1], [27, 1], [2, 1], [50, 0], [16, 1], [42, 0], [26, 0], [54, 0], [4, 1], [20, 1], [53, 0], [59, 1], [18, 0], [33, 0], [30, 0]], 
      [[36, 1], [47, 0], [35, 1], [6, 0], [29, 0], [34, 0], [13, 1], [9, 1], [12, 1], [49, 0], [32, 0], [68, 0], [67, 1], [1, 0], [10, 0], [37, 1], [14, 1], [8, 0], [61, 1], [39, 0], [38, 0], [64, 1], [65, 0], [55, 1], [46, 0], [66, 1], [52, 0], [48, 0], [51, 1], [23, 1], [15, 0], [56, 0], [31, 0], [43, 1], [60, 0]], 
      [[63, 0], [45, 1]]]

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
simulator.simulator.time_teriminate = t*0.25
ax = simulator.field.render(working_lines=False, show=False, start=False)
t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, True, True, True, False)
print(f"Simulator paused")
for idx, status in enumerate(simulator.simulator.car_status):
    print(f"Car {idx+1} | line {status['line']} | entry {status['entry']} | pos ({status['pos'][0]:3.2f}, {status['pos'][1]:3.2f}) | inline {status['inline']}")
# if len(car_cfg) <= 2:
#     veh_del = []
# elif len(car_cfg) <= 3:
#     veh_del = random.sample(range(len(car_cfg)), len(car_cfg)-1)
# else:
#     veh_del = random.sample(range(len(car_cfg)), len(car_cfg)-2)
veh_del = [0, 2]
num_veh = len(car_cfg) - len(veh_del)    
car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for idx, cur_car in enumerate(car_cfg) if idx not in veh_del])).float()

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

ac = load_model(checkpoint)
ac.sequential_sel = True
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
                    markersize=12) for start in field.starts]
t2, c2, s2, _, _, _, figs2 = simulator.simulate(ax, True, True, True, True)

print(f"Simulation result: s={np.round(s2, 2)}m, t={np.round(t2, 2)}s, c={np.round(c2, 2)}L")

figs = figs2
save_dir = os.path.join(data, algo + f"_{f}_veh.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(save_dir, fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
# map(videoWriter.write, figs)
for fig in figs:
    videoWriter.write(fig)
videoWriter.release() 
    
print(save_dir)
save_dir = os.path.join(data, algo + f"_{f}_veh.png")
plt.title(f'$s_P$={np.round(s2, 2)}m, $t_P$={np.round(t2, 2)}s, $c_P$={np.round(c2, 2)}L', fontsize=20)
plt.savefig(save_dir)
plt.close('all')
print(save_dir)

# ============================ dynamic arrangement ============================
field = load_dict(os.path.join(data, 'field.pkl'))
field.starts = ['bound-3-2' for _ in range(4)]
field.ends = ['bound-3-2' for _ in range(4)]
field.make_working_graph()
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T0, debug=True)
simulator.simulator.time_teriminate = t*0.25
ax = simulator.field.render(working_lines=False, show=False, start=False)
t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, True, True, True, False)
# if len(car_cfg) <= 2:
#     veh_del = []
# elif len(car_cfg) <= 3:
#     veh_del = random.sample(range(len(car_cfg)), len(car_cfg)-1)
# else:
#     veh_del = random.sample(range(len(car_cfg)), len(car_cfg)-2)
num_veh = len(car_cfg) - len(veh_del)    
# veh_del = [0, 2]
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
                    markersize=12) for start in field.starts]
t2, c2, s2, _, _, _, figs2 = simulator.simulate(ax, True, True, True, True)

print(f"Simulation result: s={np.round(s2, 2)}m, t={np.round(t2, 2)}s, c={np.round(c2, 2)}L")

figs = figs2
algo = 'GA'
save_dir = os.path.join(data, algo + f"_{f}_veh.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(save_dir, fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
# map(videoWriter.write, figs)
for fig in figs:
    videoWriter.write(fig)
videoWriter.release() 
    
print(save_dir)
save_dir = os.path.join(data, algo + f"_{f}_veh.png")
plt.title(f'$s_P$={np.round(s2, 2)}m, $t_P$={np.round(t2, 2)}s, $c_P$={np.round(c2, 2)}L', fontsize=20)
plt.savefig(save_dir)
plt.close('all')
print(save_dir)