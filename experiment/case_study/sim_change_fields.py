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

data = "/home/fanyx/mdvrp/experiment/case_study/4_4/0_39_24"
field_ia = load_dict(os.path.join(data, 'field.pkl'))


car_cfg = [{'vw': 2.5, 'vv': 4.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
            {'vw': 3, 'vv': 5, 'cw': 0.007, 'cv': 0.005, 'tt': 0, 'min_R': 6},
            {'vw': 3, 'vv': 5.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
            {'vw': 2, 'vv': 5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
            {'vw': 3, 'vv': 6, 'cw': 0.01, 'cv': 0.008, 'tt': 0, 'min_R': 6},]
car_cfg_0 = [{'vw': 2.5, 'vv': 4.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 0.0001},
            {'vw': 3, 'vv': 5, 'cw': 0.007, 'cv': 0.005, 'tt': 0, 'min_R': 0.0001},
            {'vw': 3, 'vv': 5.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 0.0001},
            {'vw': 2, 'vv': 5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 0.0001},
            {'vw': 3, 'vv': 6, 'cw': 0.01, 'cv': 0.008, 'tt': 0, 'min_R': 0.0001},]
algo = 'PPO-t'
f = 't'

checkpoint_ia = "/home/fanyx/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-04__12-53_t/best_model39.pt"
checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2024-12-27__12-11/best_model31.pt"

car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for cur_car in car_cfg])).float()

# ============================ MDVRP model ============================

field = multiField(num_splits=[1, 1], 
                       starts=['bound-2-1', 
                               'bound-2-1', 
                               'bound-2-1', 
                               'bound-2-1', 
                               'bound-2-1'],
                       ends=['bound-2-1',
                             'bound-2-1',
                             'bound-2-1',
                             'bound-2-1',
                             'bound-2-1']
                       )

field.from_ia(field_ia)
field_list = [0, 3]
field.make_working_graph(field_list)
field_ia.make_working_graph(field_list)

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

chosen_idx, chosen_entry = field.edit_fields(simulator.simulator.car_status)

pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
info = {'veh_key_padding_mask': torch.zeros((1, len(car_tensor))).bool(), 'num_veh': torch.tensor([[car_tensor.shape[0]]])}

with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, 
                                   force_chosen=True, 
                                   chosen_idx=chosen_idx, chosen_entry=chosen_entry)
T = decode(seq_enc[0])

print(f"Simulator restarting...")
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T, debug=True)
[ax.plot(field.Graph.nodes[start]['coord'][0], 
                    field.Graph.nodes[start]['coord'][1], 
                    '*y', 
                    markersize=20) for start in field.starts]
t2, c2, s2, _, _, _, figs2 = simulator.simulate(ax, True, True, True, False)

print(f"Simulation result: s={np.round(s1+s2, 2)}m, t={np.round(t1+t2, 2)}s, c={np.round(c1+c2, 2)}L")
plt.close('all')


figs = figs1 + figs2
if figs is not None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(os.path.join(data, algo + '_field' + ".mp4"), fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
    # map(videoWriter.write, figs)
    for fig in figs:
        videoWriter.write(fig)
    videoWriter.release() 
else:
    print('error')
    
print(os.path.join(data, algo + '_field' + ".mp4"))
plt.close('all')

# ============================ dynamic arrangement ============================
field = multiField(num_splits=[1, 1], 
                       starts=['bound-2-1', 
                               'bound-2-1', 
                               'bound-2-1', 
                               'bound-2-1', 
                               'bound-2-1'],
                       ends=['bound-2-1',
                             'bound-2-1',
                             'bound-2-1',
                             'bound-2-1',
                             'bound-2-1']
                       )

field_ia = load_dict(os.path.join(data, 'field.pkl'))
field_ia.make_working_graph(field_list)
field.from_ia(field_ia)
field.make_working_graph(field_list)

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
field_ia.make_working_graph()
chosen_idx = [[[list(field_ia.working_graph.nodes()).index(line) for line in chosen_line[idx]] for idx in range(len(car_cfg))]]
chosen_entry = [[[line[1] for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]]

for idx, status in enumerate(simulator.simulator.car_status):
    if status['inline']:
        chosen_idx[0][idx].append(list(field_ia.working_graph.nodes()).index(status['line']))
        chosen_entry[0][idx].append(status['entry'])

pygdata = from_networkx(field_ia.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
info = {'veh_key_padding_mask': torch.zeros((1, len(car_tensor))).bool(), 'num_veh': torch.tensor([[car_tensor.shape[0]]])}

with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac_ia(data_t, info, deterministic=True, criticize=False, 
                                      force_chosen=True, 
                                      chosen_idx=chosen_idx, chosen_entry=chosen_entry)

T_ia_2 = ia_util.decode(seq_enc[0])
T = [[] for _ in T_ia_2]
for idx, (a, c, status)  in enumerate(zip(T_ia_2, chosen_idx[0], simulator.simulator.car_status)):
    if status['inline']:
        T[idx] = a[len(c)-1:]
    else:
        T[idx] = a[len(c):]
print(f"Simulator restarting...")

simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T, debug=True)
[ax.plot(field.Graph.nodes[start]['coord'][0], 
                    field.Graph.nodes[start]['coord'][1], 
                    '*y', 
                    markersize=20) for start in field.starts]
t2, c2, s2, _, _, _, figs2 = simulator.simulate(ax, True, True, True, False)

print(f"Simulation result: s={np.round(s1+s2, 2)}m, t={np.round(t1+t2, 2)}s, c={np.round(c1+c2, 2)}L")

figs = figs1 + figs2
if figs is not None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(os.path.join(data, algo + '_field' + "_ia.mp4"), fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
    # map(videoWriter.write, figs)
    for fig in figs:
        videoWriter.write(fig)
    videoWriter.release() 
else:
    print('error')
    
print(os.path.join(data, algo + '_field' + "_ia.mp4"))
plt.close('all')