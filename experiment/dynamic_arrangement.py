import mdvrp.env.dubins as dubins
from shapely import geometry
import geopandas
import utm
import cv2
from geopandas import GeoSeries
import pandas as pd
import matplotlib.pyplot as plt
from mdvrp.env.field import Field
from mdvrp.env.car import car, Robot
from mdvrp.env.multi_field import multiField
from mdvrp.algo import MGGA
from mdvrp.utils import fit, load_dict, decode
from mdvrp.env.simulator import arrangeSimulator
import os
from mdvrp.model.ac import load_model
from torch_geometric.data import Batch
import torch
from torch_geometric.utils import unbatch, from_networkx
import numpy as np

data = "/home/xuht/Intelligent-Agriculture/MAdata/4_4/0_39_24"
# data = "/home/xuht/Intelligent-Agriculture/MAdata/3_4/0_39_12"
field: multiField = load_dict(os.path.join(data, 'field.pkl'))
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
algo = 'PPO-c'
f = 't'
GA = MGGA(f = f, order = True, gen_size=200, max_iter=200)
tS = np.zeros((len(car_cfg), len(field.line_length)))
for i in range(len(car_cfg)):
    tS[i] = field.line_length / car_cfg[i]['vw']

# GA = MGGA(f = 't', order = True, render = False, gen_size = 200, max_iter = 200)
# T, best, log = GA.optimize(field.D_matrix, 
#                            np.tile(field.ori, (len(car_cfg), 1, 1, 1)), 
#                            car_cfg, 
#                            field.line_length, 
#                            np.tile(field.ori, (len(car_cfg), 1, 1, 1)), 
#                            field.line_field_idx)

# checkpoint = "/home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-04__21-43_s/best_model14.pt"
# checkpoint = "/home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-05__11-05_c/best_model23.pt"
checkpoint = "/home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-04__12-53_t/best_model39.pt"
# checkpoint = "/home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_safe_rl/esb_ppo_lag/2024-03-14__14-35_L15/best_model21.pt"
# checkpoint = "/home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_safe_rl/esb_ppo_lag/2024-03-14__14-37_L20/best_model24.pt"
car_cfg_v = [[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for cur_car in car_cfg]
car_tensor = torch.tensor(np.array(car_cfg_v)).float()
ac = load_model(checkpoint)
field_list = [0, 3]
field.make_working_graph(field_list)
pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
# pygdata = load_dict(os.path.join(data, 'pygdata.pkl'))
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
info = {'veh_key_padding_mask': torch.zeros((1, len(car_tensor))).bool(), 'num_veh': torch.tensor([[car_tensor.shape[0]]])}

with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, )

T = decode(seq_enc[0])

figure, ax = plt.subplots()
ax.axis('equal')
name = 'dijkstra_path.png'
path = os.path.join(data, name)
sim = arrangeSimulator(field, car_cfg_0)
sim.init_simulation(T, init_simulator=False)
sim.render_arrange(ax)
plt.savefig(path)

print("Initializing")
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T, debug=True)
simulator.render_arrange()
name = 'dubins_path.png'
path = os.path.join(data, name)
plt.savefig(path)
# cars = [Robot({'car_model': "car3", 'working_width': working_width, **cfg}, 
#              state_ini=[path[0, 0], path[0, 1], 0, 0, f_dir, 0], debug=True) 
#         for path, f_dir, cfg in zip(paths, first_dir, car_cfg)]
# simulator = Simulator(paths, cars, working_list, max_step=500000)
# simulator.set_drive_mode("direct")#physical
print("Start simulation")
t, c, s, car_time, car_dis, figs = simulator.simulate(True, False, True)
name = 'working_result.png'
path = os.path.join(data, name)
plt.savefig(path)

chosen_idx = [[line[0] for line in simulator.simulator.line_cache[idx]] for idx in range(len(car_cfg))]
chosen_entry = [[line[1] for line in simulator.simulator.line_cache[idx]] for idx in range(len(car_cfg))]

if figs is not None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(os.path.join(data, algo + ".mp4"), fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
    # map(videoWriter.write, figs)
    for fig in figs:
        videoWriter.write(fig)
    videoWriter.release() 

# simulator.rollout(np.array([[2, 2], [2, 2], [4, 4]]), render=True, ax=ax)#
# s = 0
# c = 0
# car_time = []
# car_dis = []
# for robo in simulator.car_list:
#     s += robo.total_distance
#     c += robo.total_cost
#     car_time.append([robo.tv, robo.tw])
#     car_dis.append([robo.driving_distance, robo.working_distance])
# t = simulator.time
print(car_time)
print(car_dis)
print(np.sum(field.line_length))
t_exp = 1 / fit(field.D_matrix, 
            np.tile(field.ori, (len(car_cfg), 1, 1, 1)), 
            np.tile(field.ori, (len(car_cfg), 1, 1, 1)),
            car_cfg, 
            field.line_length,
            T,
            type = 't',
            tS_t=False)
s_exp = 1 / fit(field.D_matrix, 
            np.tile(field.ori, (len(car_cfg), 1, 1, 1)), 
            np.tile(field.ori, (len(car_cfg), 1, 1, 1)),
            car_cfg, 
            tS,
            T,
            type = 's') + np.sum(field.line_length)
c_exp = 1 / fit(field.D_matrix, 
            np.tile(field.ori, (len(car_cfg), 1, 1, 1)), 
            np.tile(field.ori, (len(car_cfg), 1, 1, 1)),
            car_cfg, 
            tS,
            T,
            type = 'c')

print(f'Time expected: {np.round(t_exp, 2)} s, real time: {np.round(t, 2)} s. ')
print(f'Distance expected: {np.round(s_exp, 2)} m, real distance: {np.round(s, 2)} m. ')
print(f'Energy cost expected: {np.round(c_exp, 2)} L, real energy cost: {np.round(c, 2)} L. ')
print(f'Max e_n: {np.round(np.max([robo.total_ect for robo in simulator.simulator.car_list])/simulator.simulator.steps, 2)} m. ')
print(f'Max e_t: {np.round(np.max([robo.total_ev for robo in simulator.simulator.car_list])/simulator.simulator.steps, 2)} m/s. ')
# plt.show()