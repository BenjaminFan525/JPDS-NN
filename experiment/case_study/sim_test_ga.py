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

data = "/home/fanyx/mdvrp/data/Gdataset/Task_test_multifield/single_depot/4_4/10_88_12"
field = load_dict(os.path.join(data, 'field.pkl'))

car_cfg = load_dict(os.path.join(data, 'car_cfg.pkl'))

algo = 'PPO-t'
f = 't'

checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2025-02-02__08-36__t/best_model38.pt"

car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for cur_car in car_cfg])).float()

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
# field_list = [0, 3]
# field.make_working_graph(field_list)
# # field.make_working_graph()

# ============================ MDVRP model ============================
print("MDVRP model initializing...")
ac = load_model(checkpoint)
ac.sequential_sel = True

pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
info = {'veh_key_padding_mask': torch.zeros((1, len(car_tensor))).bool(), 'num_veh': torch.tensor([[car_tensor.shape[0]]])}

with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, )
T = decode(seq_enc[0])

# print(f"Simulator initializng...")
# simulator = arrangeSimulator(field, car_cfg)
# simulator.init_simulation(T)
# ax = simulator.field.render(working_lines=False, show=False, start=False)
# t1, c1, s1, car_time, car_dis, ax, figs = simulator.simulate(ax, True, True, True, False)
# print(f"Simulation result: s={np.round(s1, 2)}m, t={np.round(t1, 2)}s, c={np.round(c1, 2)}L")
 
# s, t, c = fit(field.D_matrix, 
#         field.ori.transpose(0, 1, 3, 2), 
#         field.des.transpose(0, 1, 3, 2),
#         car_cfg, 
#         field.line_length,
#         T, 
#         tS_t=False,
#         type='all')

# print(f"Fit result: s={np.round(s, 2)}m, t={np.round(t, 2)}s, c={np.round(c, 2)}L")
# save_dir = os.path.join(data, algo + "_mdvrp.mp4")
# if figs is not None:
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     videoWriter = cv2.VideoWriter(save_dir, fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
#     # map(videoWriter.write, figs)
#     for fig in figs:
#         videoWriter.write(fig)
#     videoWriter.release() 
# else:
#     print('error')
    
# print(save_dir)

# ============================ GA ============================
print("GA initializing...")
GA = MGGA(f = f, order = True, gen_size=100, max_iter=100)
T, best, log = GA.optimize(field.D_matrix, 
                field.ori, 
                car_cfg, 
                field.line_length, 
                field.des,
                field.line_field_idx,) 

simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T)
ax = simulator.field.render(working_lines=False, show=False, start=False)
t1, c1, s1, car_time, car_dis, ax, figs = simulator.simulate(ax, True, True, True, False)
print(f"Simulation result: s={np.round(s1, 2)}m, t={np.round(t1, 2)}s, c={np.round(c1, 2)}L")

save_dir = os.path.join(data, algo + "_ga.mp4")
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