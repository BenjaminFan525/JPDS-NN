from matplotlib.hatch import VerticalHatch
from env.multi_field import multiField
from utils import fit, load_dict, decode
import os
from torch_geometric.data import Batch
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
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


data = "/home/fanyx/mdvrp/data/Gdataset/Task_test_multifield/single_depot/4_4/60_77_12"
field = load_dict(os.path.join(data, 'field.pkl'))

car_cfg = [{'vw': 1.5, 'vv': 3.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 5.5},
            {'vw': 2, 'vv': 4, 'cw': 0.007, 'cv': 0.005, 'tt': 0, 'min_R': 5.5},
            {'vw': 2.7, 'vv': 4.7, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 5.5},
            {'vw': 2.2, 'vv': 4.2, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 5.5},
            {'vw': 1.8, 'vv': 3.8, 'cw': 0.007, 'cv': 0.005, 'tt': 0, 'min_R': 5.5},]

checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2025-02-02__08-36__t/best_model38.pt"
ac = load_model(checkpoint)
ac.sequential_sel = True

veh_sel = [0, 1]
depot = 'bound-3-2'
car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for idx, cur_car in enumerate(car_cfg) if idx in veh_sel])).float()
field.starts = [depot for _ in veh_sel]
field.ends = [depot for _ in veh_sel]
field.num_veh = len(field.starts)
field.num_endpoints = 2*field.num_veh
field.make_working_graph()
num_veh = car_tensor.shape[0]
veh_key_padding_mask = torch.zeros((1, len(car_tensor))).bool()
pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
info = {'veh_key_padding_mask': veh_key_padding_mask, 'num_veh': torch.tensor([[num_veh]])}

with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, )
T = decode(seq_enc[0])

T_full = []
t_idx = 0
for idx in range(len(car_cfg)):
    if idx in veh_sel:
        T_full.append(T[t_idx])
        t_idx += 1
    else:
        T_full.append([])

field.starts = [depot for idx in range(len(car_cfg))]
field.ends = [depot for idx in range(len(car_cfg))]
field.num_veh = len(field.starts)
field.num_endpoints = 2*field.num_veh
field.make_working_graph()
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T_full)
ax = field.render(working_lines=False, show=False, start=True, end=False)
ax.set_title(f'$s_P$={0}m, $t_P$={0}s, $c_P$={0}L', fontsize=20)
t1, c1, s1, _, _, _, figs1 = simulator.simulate(ax, True, True, True, False)
ax.set_title(f'$s_P$={np.round(s1, 2)}m, $t_P$={np.round(t1, 2)}s, $c_P$={np.round(c1, 2)}L', fontsize=20)
print(f"Simulation result: s={np.round(s1, 2)}m, t={np.round(t1, 2)}s, c={np.round(c1, 2)}L")

veh_sel = [2]
depot = 'bound-3-2'
car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for idx, cur_car in enumerate(car_cfg) if idx in veh_sel])).float()
field.starts = [depot for _ in veh_sel]
field.ends = [depot for _ in veh_sel]
field.num_veh = len(field.starts)
field.num_endpoints = 2*field.num_veh
field.make_working_graph()
num_veh = car_tensor.shape[0]
veh_key_padding_mask = torch.zeros((1, len(car_tensor))).bool()
pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
info = {'veh_key_padding_mask': veh_key_padding_mask, 'num_veh': torch.tensor([[num_veh]])}

with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, )
T = decode(seq_enc[0])

T_full = []
t_idx = 0
for idx in range(len(car_cfg)):
    if idx in veh_sel:
        T_full.append(T[t_idx])
        t_idx += 1
    else:
        T_full.append([])

field.starts = [depot for idx in range(len(car_cfg))]
field.ends = [depot for idx in range(len(car_cfg))]
field.num_veh = len(field.starts)
field.num_endpoints = 2*field.num_veh
field.make_working_graph()
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T_full)
# ax = field.render(working_lines=False, show=False, start=True, end=False)
t2, c2, s2, _, _, _, figs2 = simulator.simulate(ax, True, True, True, False)
ax.set_title(f'$s_P$={np.round(s1+s2, 2)}m, $t_P$={np.round(t1+t2, 2)}s, $c_P$={np.round(c1+c2, 2)}L', fontsize=20)
print(f"Simulation result: s={np.round(s1+s2, 2)}m, t={np.round(t1+t2, 2)}s, c={np.round(c1+c2, 2)}L")

veh_sel = [3, 4]
depot = 'bound-3-2'
car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for idx, cur_car in enumerate(car_cfg) if idx in veh_sel])).float()
field.starts = [depot for _ in veh_sel]
field.ends = [depot for _ in veh_sel]
field.num_veh = len(field.starts)
field.num_endpoints = 2*field.num_veh
field.make_working_graph()
num_veh = car_tensor.shape[0]
veh_key_padding_mask = torch.zeros((1, len(car_tensor))).bool()
pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
info = {'veh_key_padding_mask': veh_key_padding_mask, 'num_veh': torch.tensor([[num_veh]])}

with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, )
T = decode(seq_enc[0])

T_full = []
t_idx = 0
for idx in range(len(car_cfg)):
    if idx in veh_sel:
        T_full.append(T[t_idx])
        t_idx += 1
    else:
        T_full.append([])

field.starts = [depot for idx in range(len(car_cfg))]
field.ends = [depot for idx in range(len(car_cfg))]
field.num_veh = len(field.starts)
field.num_endpoints = 2*field.num_veh
field.make_working_graph()
simulator = arrangeSimulator(field, car_cfg)
simulator.init_simulation(T_full)
# ax = field.render(working_lines=False, show=False, start=True, end=False)
t3, c3, s3, _, _, _, figs3 = simulator.simulate(ax, True, True, True, False)

print(f"Simulation result: s={np.round(s1+s2+s3, 2)}m, t={np.round(t1+t2+t3, 2)}s, c={np.round(c1+c2+c3, 2)}L")

figs = figs1 + figs2 + figs3

ax.set_title(f'$s_P$={np.round(s1+s2+s3, 2)}m, $t_P$={np.round(t1+t2+t3, 2)}s, $c_P$={np.round(c1+c2+c3, 2)}L', fontsize=20)
simulator.simulator.render(ax, show = False, label=False)
canvas = FigureCanvasAgg(plt.gcf())
w, h = canvas.get_width_height()
canvas.draw()
buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
buf.shape = (w, h, 3)
buf = buf[:, :, [2, 1, 0]]
# buf = np.roll(buf, 3, axis=2)
image = Image.frombytes("RGB", (w, h), buf.tobytes())
figs += [np.asarray(image)[:, :, :3]]*10

case = 'sequential'
save_dir = os.path.join(data, case + ".mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(save_dir, fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
# map(videoWriter.write, figs)
for fig in figs:
    videoWriter.write(fig)
videoWriter.release() 
    
print(save_dir)