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

car_cfg_0 = [{'vw': 1, 'vv': 5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
            {'vw': 1, 'vv': 5, 'cw': 0.007, 'cv': 0.005, 'tt': 0, 'min_R': 6},
            {'vw': 1, 'vv': 5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
            {'vw': 1, 'vv': 5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},]
algo = 'PPO'
f = 't'
checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2025-02-02__08-36__t/best_model38.pt"
# f = 's'
# checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2025-01-31__10-37__s/best_model19.pt"
# f = 'c'
# checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2025-01-31__10-38__c/best_model22.pt"

car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for cur_car in car_cfg])).float()

def random_arrangement(num_car, num_target):
    gen = {'tar': [[x, int(np.round(random.random()))] for x in range(num_target)], 'split': np.random.choice(range(num_target + 1), num_car - 1, replace = True).tolist()}
    random.shuffle(gen['tar'])
    gen['split'].sort()
    return ia_util.decode(gen)

# T0 = random_arrangement(field.ori.shape[:2][0], field.ori.shape[:2][1])

# print(T0)

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

# ============================ MDVRP model ============================
print(f"Simulator initializng...")
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T0, debug=True)

simulator.simulator.time_teriminate = t*0.25
ax = simulator.field.render(working_lines=False, show=False, start=False)
t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, True, True, True, False)
print(f"Simulator paused")
for idx, status in enumerate(simulator.simulator.car_status):
    print(f"Car {idx+1} | line {status['line']} | entry {status['entry']} | pos ({status['pos'][0]:3.2f}, {status['pos'][1]:3.2f}) | inline {status['inline']}")

chosen_idx, chosen_entry = field.edit_fields(simulator.simulator.car_status)
ac = load_model(checkpoint)
ac.sequential_sel = True
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
                    markersize=12) for start in field.starts]
t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, True, True, True, True)

print(f"Simulation result: s={np.round(s2, 2)}m, t={np.round(t2, 2)}s, c={np.round(c2, 2)}L")

figs = figs2
save_dir = os.path.join(data, algo + f"_{f}_rearrange.mp4")
if figs is not None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(save_dir, fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
    # map(videoWriter.write, figs)
    for fig in figs:
        videoWriter.write(fig)
    videoWriter.release() 
else:
    print('error')
    
print(save_dir)
save_dir = os.path.join(data, algo + f"_{f}_rec.png")
plt.title(f'$s_P$={np.round(s2, 2)}m, $t_P$={np.round(t2, 2)}s, $c_P$={np.round(c2, 2)}L', fontsize=20)
plt.savefig(save_dir)
plt.close('all')
print(save_dir)

# ============================ GA ============================
field = load_dict(os.path.join(data, 'field.pkl'))
field.starts = ['bound-3-2' for _ in range(4)]
field.ends = ['bound-3-2' for _ in range(4)]
field.make_working_graph()
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T0, debug=True)

simulator.simulator.time_teriminate = t*0.25
ax = simulator.field.render(working_lines=False, show=False, start=False)
t1, c1, s1, car_time, car_dis, ax, figs1 = simulator.simulate(ax, True, True, True, False)
chosen_idx, chosen_entry = field.edit_fields(simulator.simulator.car_status)

GA = MGGA(f = f, order = True, gen_size=100, max_iter=100)
T, best, log = GA.optimize(field.D_matrix, 
                field.ori, 
                car_cfg, 
                field.line_length, 
                field.des,
                field.line_field_idx,
                force_chosen=True,chosen_entry=chosen_entry[0], chosen_idx=chosen_idx[0])

print(f"Simulator initializng...")
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T)
[ax.plot(field.Graph.nodes[start]['coord'][0], 
                    field.Graph.nodes[start]['coord'][1], 
                    '*y', 
                    markersize=12) for start in field.starts]
t2, c2, s2, car_time, car_dis, ax, figs2 = simulator.simulate(ax, True, True, True, True)
print(f"Simulation result: s={np.round(s2, 2)}m, t={np.round(t2, 2)}s, c={np.round(c2, 2)}L")

figs = figs2
algo = 'GA'
save_dir = os.path.join(data, algo + f"_{f}_rearrange.mp4")
if figs is not None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(save_dir, fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
    # map(videoWriter.write, figs)
    for fig in figs:
        videoWriter.write(fig)
    videoWriter.release() 
else:
    print('error')
    
print(save_dir)

save_dir = os.path.join(data, algo + f"_{f}_rec.png")
plt.title(f'$s_P$={np.round(s2, 2)}m, $t_P$={np.round(t2, 2)}s, $c_P$={np.round(c2, 2)}L', fontsize=20)
plt.savefig(save_dir)
print(save_dir)