import env.dubins as dubins
from shapely import geometry
import geopandas
import utm
from geopandas import GeoSeries
import pandas as pd

from env.field import Field
from env.car import car, Robot
from env.multi_field import multiField
from algo import MGGA
from utils import fit, load_dict, decode
import os
from model.ac import load_model
from copy import deepcopy
from torch_geometric.data import Batch
import torch
from torch_geometric.utils import unbatch, from_networkx
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from env import arrangeSimulator, simulator
import cv2


data = "/data/fanyx_files/mdvrp/experiment/case_study/4_4/8_52_12"
field = load_dict(os.path.join(data, 'field.pkl'))

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

checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2024-12-27__12-11/best_model31.pt"

car_cfg_v = [[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for cur_car in car_cfg]
car_tensor = torch.tensor(np.array(car_cfg_v)).float()

field.starts=['bound-2-1', 
              'bound-2-1', 
              'bound-2-1', 
              'bound-2-1', 
              'bound-2-1']

field.ends=['bound-2-1',
            'bound-2-1',
            'bound-2-1',
            'bound-2-1',
            'bound-2-1']

# field = multiField(num_splits=[1, 1], 
#                    starts=['bound-2-1', 
#                           'bound-1-3', 
#                           'bound-3-2', 
#                           'bound-0-0', 
#                           'bound-0-0'])
# field = multiField([1, 1], homes=['bound-2-1'])

# field.from_ia(field_ia)
# field_list = [0, 3]
# field.make_working_graph(field_list)
field.make_working_graph()

ac = load_model(checkpoint)
ac.num_veh = field.num_veh
ac.sequential_sel = True
ac.end_en = True if field.ends is not None else False
ac.encoder.end_en = ac.end_en

pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
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
t, c, s, car_time, car_dis, figs = simulator.simulate(True, True, True)
# plt.show()
name = 'working_result.png'
path = os.path.join(data, name)
plt.savefig(path)
print(path)

chosen_idx = [[line[0] for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]
chosen_entry = [[line[1] for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]

if figs is not None:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(os.path.join(data, algo + ".mp4"), fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
    # map(videoWriter.write, figs)
    for fig in figs:
        videoWriter.write(fig)
    videoWriter.release() 
else:
    print('error')
    
print(os.path.join(data, algo + ".mp4"))