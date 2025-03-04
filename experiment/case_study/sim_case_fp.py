from matplotlib.hatch import VerticalHatch
from env.multi_field import multiField
from utils import fit, load_dict, decode
import os
from torch_geometric.data import Batch
import torch
from torch_geometric.utils import unbatch, from_networkx
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
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


def arrangement(veh_sel, field_sel, start, end, car_cfg, field, ac):
    car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                    cur_car['cw'], cur_car['cv'],
                    cur_car['tt']] for idx, cur_car in enumerate(car_cfg) if idx in veh_sel])).float()
    field.starts = [start for _ in veh_sel]
    field.ends = [end for _ in veh_sel]
    field.num_veh = len(field.starts)
    field.num_endpoints = 2*field.num_veh
    field.make_working_graph(field_sel)
    num_veh = car_tensor.shape[0]
    veh_key_padding_mask = torch.zeros((1, len(car_tensor))).bool()
    pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
    data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
    info = {'veh_key_padding_mask': veh_key_padding_mask, 'num_veh': torch.tensor([[num_veh]])}

    with torch.no_grad():
        seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, )
    T = decode(seq_enc[0])

    # T_full = []
    # t_idx = 0
    # for idx in range(len(car_cfg)):
    #     if idx in veh_sel:
    #         T_full.append(T[t_idx])
    #         t_idx += 1
    #     else:
    #         T_full.append([])

    for veh in T:
        for a in veh:
            a[0] = field.working_line_list[a[0]]

    return T

data = "/home/fanyx/mdvrp/data/Gdataset/Task_test_multifield/single_depot/4_4/60_77_12"

car_cfg = [{'vw': 1.5, 'vv': 3.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 5.5},
            {'vw': 2, 'vv': 4, 'cw': 0.007, 'cv': 0.005, 'tt': 0, 'min_R': 5.5},
            {'vw': 2.7, 'vv': 4.7, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 5.5},
            {'vw': 2.2, 'vv': 4.2, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 5.5},
            {'vw': 1.8, 'vv': 3.8, 'cw': 0.007, 'cv': 0.005, 'tt': 0, 'min_R': 5.5},]

checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2025-02-02__08-36__t/best_model38.pt"
ac = load_model(checkpoint)
ac.sequential_sel = True
print(ac)

field = load_dict(os.path.join("/home/fanyx/mdvrp/data/Gdataset/Task_1depot/Test/4_1/6_82_24", 'field.pkl'))
ax = field.render(show=False, entry_point=True, boundary=True, line_colors='k', start=False, end=False)
plt.savefig("field.png")

starts = ['bound-3-2', 'bound-3-0', 'bound-2-0', 'bound-1-0']
ends = ['bound-3-0', 'bound-2-0', 'bound-1-0', 'bound-3-2']
field_starts = ['bound-3-2' for _ in range(5)]
field_ends = ['bound-3-2' for _ in range(5)]
field_ids = [3, 2, 0, 1]
figs = []
s, t, c = 0, 0, 0

field = load_dict(os.path.join(data, 'field.pkl'))
ax = field.render(working_lines=False, show=False, start=False, end=False)
ax.plot(field.Graph.nodes[starts[0]]['coord'][0], 
                field.Graph.nodes[starts[0]]['coord'][1], 
                '*y', 
                markersize=20)
ax.set_title(f'$s_P$={np.round(s, 2)}m, $t_P$={np.round(t, 2)}s, $c_P$={np.round(c, 2)}L', fontsize=20)

for idx in range(6):
    T = []
    if idx < 4:
        field = load_dict(os.path.join(data, 'field.pkl'))
        T += arrangement([0, 1], [field_ids[idx]], starts[idx], ends[idx], car_cfg, field, ac)
        field_starts[0] = starts[idx]
        field_starts[1] = starts[idx]
        field_ends[0] = ends[idx]
        field_ends[1] = ends[idx]
    else:
        T += [[], []]
        field_starts[0] = starts[0]
        field_starts[1] = starts[0]
        field_ends[0] = ends[-1]
        field_ends[1] = ends[-1]

    if idx > 0 and idx < 5:
        field = load_dict(os.path.join(data, 'field.pkl'))
        T += arrangement([2], [field_ids[idx-1]], starts[idx-1], ends[idx-1], car_cfg, field, ac)
        field_starts[2] = starts[idx-1]
        field_ends[2] = ends[idx-1]
    else:
        T += [[]]
        field_starts[2] = starts[0]
        field_ends[2] = ends[-1]        

    if idx > 1:
        field = load_dict(os.path.join(data, 'field.pkl'))
        T += arrangement([3, 4], [field_ids[idx-2]], starts[idx-2], ends[idx-2], car_cfg, field, ac)
        field_starts[3] = starts[idx-2]
        field_starts[4] = starts[idx-2]
        field_ends[3] = ends[idx-2]
        field_ends[4] = ends[idx-2]
    else:
        T += [[], []]
        field_starts[3] = starts[0]
        field_starts[4] = starts[0]
        field_ends[3] = ends[-1]
        field_ends[4] = ends[-1]
    
    field = load_dict(os.path.join(data, 'field.pkl'))
    for veh in T:
        for a in veh:
            a[0] = field.working_line_list.index(a[0])
    
    field.starts = field_starts
    field.ends = field_ends
    field.num_veh = len(field.starts)
    field.num_endpoints = 2*field.num_veh
    field.make_working_graph()
    simulator = arrangeSimulator(field, car_cfg)
    simulator.init_simulation(T)
    t0, c0, s0, _, _, _, figs0 = simulator.simulate(ax, True, True, True, False)
    s += s0
    t += t0
    c += c0
    figs += figs0

    ax.set_title(f'$s_P$={np.round(s, 2)}m, $t_P$={np.round(t, 2)}s, $c_P$={np.round(c, 2)}L', fontsize=20)
    print(f"Simulation result: s={np.round(s, 2)}m, t={np.round(t, 2)}s, c={np.round(c, 2)}L")

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

case = 'field_parallel'
save_dir = os.path.join(data, case + ".mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(save_dir, fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
# map(videoWriter.write, figs)
for fig in figs:
    videoWriter.write(fig)
videoWriter.release() 
    
print(save_dir)