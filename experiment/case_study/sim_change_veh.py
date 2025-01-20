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
data = "/home/fanyx/mdvrp/data/Gdataset/Task_test_debug/6_6/7_29_24"
# data = "/home/fanyx/mdvrp/experiment/case_study/4_4/0_39_24"
# field_ia = load_dict(os.path.join(data, 'field.pkl'))

field_ia = load_dict(os.path.join(data, 'field_ia.pkl'))
field = load_dict(os.path.join(data, 'field.pkl'))

car_cfg = load_dict(os.path.join(data, 'car_cfg.pkl'))
# car_cfg = [{'vw': 2.5, 'vv': 4.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
#             {'vw': 3, 'vv': 5, 'cw': 0.007, 'cv': 0.005, 'tt': 0, 'min_R': 6},
#             {'vw': 3, 'vv': 5.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
#             {'vw': 2, 'vv': 5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
#             {'vw': 3, 'vv': 6, 'cw': 0.01, 'cv': 0.008, 'tt': 0, 'min_R': 6},]

algo = 'PPO-t'
f = 't'

checkpoint_ia = "/home/fanyx/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-04__12-53_t/best_model39.pt"
checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2024-12-27__12-11/best_model31.pt"

car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for cur_car in car_cfg])).float()

# ============================ MDVRP model ============================

# field = multiField(num_splits=[1, 1], 
#                        starts=['bound-2-1', 
#                                'bound-2-1', 
#                                'bound-2-1', 
#                                'bound-2-1', 
#                                'bound-2-1'],
#                        ends=['bound-2-1',
#                              'bound-2-1',
#                              'bound-2-1',
#                              'bound-2-1',
#                              'bound-2-1']
#                        )

# field.from_ia(field_ia)
# field.make_working_graph()

ac = load_model(checkpoint)
ac.sequential_sel = True

pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
veh_key_padding_mask = torch.zeros((1, len(car_tensor))).bool()
info = {'veh_key_padding_mask': veh_key_padding_mask, 'num_veh': torch.tensor([[car_tensor.shape[0]]])}

with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, )
T = decode(seq_enc[0])

simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T, debug=True)
t, c, s, _, _, _, _ = simulator.simulate(None, False, False, False, False)
print(f"MDVRP result: s={np.round(s, 2)}m, t={np.round(t, 2)}s, c={np.round(c, 2)}L")

ac_ia = load_model(checkpoint_ia, model_type='ia')
pygdata = from_networkx(field_ia.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
info = {'veh_key_padding_mask': torch.zeros((1, len(car_tensor))).bool(), 'num_veh': torch.tensor([[car_tensor.shape[0]]])}
with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac_ia(data_t, info, deterministic=True, criticize=False, )

T_ia = ia_util.decode(seq_enc[0])

simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T_ia, debug=True)
t_ia, c_ia, s_ia, _, _, _, _ = simulator.simulate(None, False, False, False, False)
print(f"IA result: s={np.round(s_ia, 2)}m, t={np.round(t_ia, 2)}s, c={np.round(c_ia, 2)}L")

if t < t_ia:
    T0 = T
    t0 = t
else:
    T0 = T_ia
    t0 = t_ia

# ============================ dynamic arrangement ============================

print(f"Simulator initializng...")
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T0, debug=True)
simulator.simulator.time_teriminate = t0*0.5
ax = simulator.field.render(working_lines=False, show=False, start=False)
t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, True, True, True, True)
print(f"Simulator paused")
for idx, status in enumerate(simulator.simulator.car_status):
    print(f"Car {idx+1} | line {status['line']} | entry {status['entry']} | pos ({status['pos'][0]:3.2f}, {status['pos'][1]:3.2f}) | inline {status['inline']}")

veh_del = [0, 1, 2, 3]
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
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(os.path.join(data, algo + '_veh' + ".mp4"), fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
# map(videoWriter.write, figs)
for fig in figs:
    videoWriter.write(fig)
videoWriter.release() 
    
print(os.path.join(data, algo + '_veh' + ".mp4"))
plt.close('all')

# ============================ dynamic arrangement ============================
# field = multiField(num_splits=[1, 1], 
#                        starts=['bound-2-1', 
#                                'bound-2-1', 
#                                'bound-2-1', 
#                                'bound-2-1', 
#                                'bound-2-1'],
#                        ends=['bound-2-1',
#                              'bound-2-1',
#                              'bound-2-1',
#                              'bound-2-1',
#                              'bound-2-1']
#                        )
# field_ia = load_dict(os.path.join(data, 'field.pkl'))
# field.from_ia(field_ia)
field_ia = load_dict(os.path.join(data, 'field_ia.pkl'))
field = load_dict(os.path.join(data, 'field.pkl'))

print(f"Simulator initializng...")
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T0, debug=True)
simulator.simulator.time_teriminate = t0*0.5
ax = simulator.field.render(working_lines=False, show=False, start=False)
t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, True, True, True, True)
print(f"Simulator paused")
for idx, status in enumerate(simulator.simulator.car_status):
    print(f"Car {idx+1} | line {status['line']} | entry {status['entry']} | pos ({status['pos'][0]:3.2f}, {status['pos'][1]:3.2f}) | inline {status['inline']}")

chosen_line = [[field_ia.working_line_list[line[0]] for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]
field.edit_fields(simulator.simulator.car_status, delete_lines=False)
chosen_idx_full = [[[list(field_ia.working_graph.nodes()).index(line) for line in chosen_line[idx]] for idx in range(len(car_cfg))]]
chosen_entry_full = [[[line[1] for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]]

for idx, status in enumerate(simulator.simulator.car_status):
    if status['inline']:
        chosen_idx_full[0][idx].append(list(field_ia.working_graph.nodes()).index(status['line']))
        chosen_entry_full[0][idx].append(status['entry'])

veh_del = [0, 1, 2, 3]
car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for idx, cur_car in enumerate(car_cfg) if idx not in veh_del])).float()
num_veh = car_tensor.shape[0]
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

pygdata = from_networkx(field_ia.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
info = {'veh_key_padding_mask': veh_key_padding_mask, 'num_veh': torch.tensor([[num_veh]])}

with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac_ia(data_t, info, deterministic=True, criticize=False, 
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
field.make_working_graph()

print(f"Simulator restarting...")

simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T_full, debug=True)
[ax.plot(field.Graph.nodes[start]['coord'][0], 
                    field.Graph.nodes[start]['coord'][1], 
                    '*y', 
                    markersize=20) for start in field.starts]
t2, c2, s2, _, _, _, figs2 = simulator.simulate(ax, True, True, True, False)

print(f"Simulation result: s={np.round(s1+s2, 2)}m, t={np.round(t1+t2, 2)}s, c={np.round(c1+c2, 2)}L")

figs = figs1 + figs2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(os.path.join(data, algo + '_veh' + "_ia.mp4"), fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
# map(videoWriter.write, figs)
for fig in figs:
    videoWriter.write(fig)
videoWriter.release() 
    
print(os.path.join(data, algo + '_veh'  + "_ia.mp4"))
plt.close('all')