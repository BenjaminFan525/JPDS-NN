from copy import deepcopy
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
import networkx as nx
import utils.common as ucommon
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

def find_nearest_points(G: nx.Graph, point: np.ndarray, num: int = 1):
    # 获取图中所有节点的坐标
    node_positions = {node: np.array(pos) for node, pos in G.nodes(data='coord') if 'b' in node}            # 计算外部点到图中每个节点的欧几里得距离
    distances = {node: np.linalg.norm(point - pos) for node, pos in node_positions.items()}

    return sorted(distances, key=distances.get)[:num]

def find_point(G: nx.Graph, point: np.ndarray):
    # 获取图中所有节点的坐标
    for node, pos in G.nodes(data='coord'):
        if np.allclose(pos, point):
            return node
    return None

def edit_fields(f, car_status: list, new_ends: list = None, fields_sel:list = 'all', veh_sel:list = [], masked_lines: list = [], delete_lines:bool = True):
    deleted_lines = [[] for _ in range(f.num_fields)]
    start_lines = []
    for (idx, status) in zip(veh_sel, car_status):
        for line in status['traveled']:
            field_idx, line_idx = int(line.split('-')[1]), int(line.split('-')[2])
            deleted_lines[field_idx].append(line_idx)
        
        if 'start' not in status['line'] and 'end' not in status['line']:
            field_idx, line_idx = int(status['line'].split('-')[1]), int(status['line'].split('-')[2])
        else:
            start_lines.append(None)
            continue
        # 作业行内点：将停止点作为新端点
        if status['inline']:
            if status['entry'] == 0:
                start_lines.append([np.array(f.fields[field_idx].working_lines[line_idx][2:4]),
                                    np.array(f.fields[field_idx].working_lines[line_idx][4:6])])
            else:
                start_lines.append([np.array(f.fields[field_idx].working_lines[line_idx][4:6]),
                                    np.array(f.fields[field_idx].working_lines[line_idx][2:4])])
        else:
            start_lines.append(None)
    
    for line in masked_lines:
        field_idx, line_idx = int(line.split('-')[1]), int(line.split('-')[2])
        if line_idx not in deleted_lines[field_idx]:
            deleted_lines[field_idx].append(line_idx)
    
    if delete_lines:
        for node in list(f.Graph.nodes()):
            if 'line' in node:
                f.Graph.remove_node(node)
        
        for delete, field in zip(deleted_lines, f.fields):
            field.working_lines = [line for idx, line in enumerate(field.working_lines) if idx not in delete] 
            field.generate_graph()

        f.Graph= nx.compose_all([f.Graph] + [field.Graph for field in f.fields])

    for (idx, status) in zip(veh_sel, car_status):
        if status['inline']:
            entry, exit = find_point(f.Graph, start_lines[idx-veh_sel[0]][0]), find_point(f.Graph, start_lines[idx-veh_sel[0]][1])
            f.Graph.add_nodes_from([(f'start-{idx}', {'coord': status['pos']})])
            f.starts[idx] = f'start-{idx}'
            f.Graph.add_weighted_edges_from([f.gen_edge(entry, f'start-{idx}')])
            field_idx, line_idx = int(entry.split('-')[1]), int(entry.split('-')[2])
            start_line = ucommon.gen_node('line', field_idx, line_idx)
            length = ucommon.get_euler_dis(f.Graph, f'start-{idx}', exit)
            f.fields[field_idx].working_lines[line_idx][1] = length
            if status['entry'] == 0:
                f.fields[field_idx].working_lines[line_idx][2:4] = [status['pos'][0], status['pos'][1]]
            else:
                f.fields[field_idx].working_lines[line_idx][4:6] = [status['pos'][0], status['pos'][1]]
            nx.set_node_attributes(f.fields[field_idx].working_graph, 
            {start_line:
                {'length': length,
                f'end{status["entry"]}': f'start-{idx}',
                'embed': torch.Tensor([np.sin(f.fields[field_idx].direction_angle), 
                                        np.cos(f.fields[field_idx].direction_angle), 
                                        length])
                }})
            start_lines[idx-veh_sel[0]] = start_line
        elif 'start' not in status['line']:
            f.Graph.add_nodes_from([(f'start-{idx}', {'coord': status['pos']})])
            f.starts[idx] = f'start-{idx}'
            for point in find_nearest_points(f.Graph, status['pos'], 2):
                f.Graph.add_weighted_edges_from([f.gen_edge(f'start-{idx}', point)])

    f.starts = [f.starts[i] for i in range(len(f.starts)) if i in veh_sel]
    f.ends = new_ends if new_ends else f.ends
    f.num_veh = len(f.starts)
    f.num_endpoints = 2*f.num_veh

    f.make_working_graph(fields_sel)

    chosen_idx, chosen_entry = [], []
    for status, start_line in zip(car_status, start_lines):
        if status['inline']:
            chosen_idx.append([list(f.working_graph.nodes()).index(start_line)])
            chosen_entry.append([status['entry']])
        else:
            chosen_idx.append([])
            chosen_entry.append([])

    return [chosen_idx], [chosen_entry]

def arrangement(veh_sel, field_sel, car_status, end, car_cfg, field, ac, unfinished):
    ori_field = deepcopy(field)
    if len(car_status): 
        traveled_lines = []
        target_lines = []
        for fidx in field_sel:
            target_lines += list(field.fields[fidx].working_graph.nodes)

        if veh_sel == [2]:
            traveled_lines += car_status[0]['traveled']
            traveled_lines += car_status[1]['traveled']
        elif veh_sel == [3, 4]:
            traveled_lines += car_status[2]['traveled']

        if len(traveled_lines):
            masked_lines = [line for line in target_lines if line not in traveled_lines]
        else:
            masked_lines = []
        chosen_idx, chosen_entry = edit_fields(field, [car_status[i] for i in veh_sel], [end for _ in veh_sel], 
                                               field_sel, veh_sel, masked_lines)
        
        for idx, vidx in enumerate(veh_sel):
            if unfinished[vidx]:
                for a in unfinished[vidx]:
                    field_idx, line_idx = int(a[0].split('-')[1]), int(a[0].split('-')[2])
                    node0 = ori_field.Graph.nodes[f"line_0-{field_idx}-{line_idx}"]['coord']
                    node1 = ori_field.Graph.nodes[f"line_1-{field_idx}-{line_idx}"]['coord']
                    node0, node1 = find_point(field.Graph, node0), find_point(field.Graph, node1)
                    if node0:
                        field_idx, line_idx = int(node0.split('-')[1]), int(node0.split('-')[2])
                    else:
                        field_idx, line_idx = int(node1.split('-')[1]), int(node1.split('-')[2])
                    a[0] = f'line-{field_idx}-{line_idx}'
                    chosen_idx[0][idx].append(list(field.working_graph.nodes()).index(a[0]))
                    chosen_entry[0][idx].append(a[1])
        
    else:
        field.starts = [starts[0] for _ in veh_sel]
        field.ends = [end for _ in veh_sel]
        field.num_veh = len(field.starts)
        field.num_endpoints = 2*field.num_veh
        field.make_working_graph(field_sel)
    
    car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                    cur_car['cw'], cur_car['cv'],
                    cur_car['tt']] for idx, cur_car in enumerate(car_cfg) if idx in veh_sel])).float()

    num_veh = car_tensor.shape[0]
    veh_key_padding_mask = torch.zeros((1, len(car_tensor))).bool()
    # if 'edge_embed' not in field.working_graph.edges[list(field.working_graph.edges)[0]]:
    #     return [[] for _ in veh_sel]
    if len(field.working_line_list) == 0:
        return [[] for _ in veh_sel]
    pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
    data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
    info = {'veh_key_padding_mask': veh_key_padding_mask, 'num_veh': torch.tensor([[num_veh]])}
    
    if len(car_status):
        with torch.no_grad():
            seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, 
                                        force_chosen=True, 
                                        chosen_idx=chosen_idx, chosen_entry=chosen_entry)
    else:
        with torch.no_grad():
            seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False)

    T = decode(seq_enc[0])

    for veh in T:
        for a in veh:
            line = field.working_line_list[a[0]]
            field_idx, line_idx = int(line.split('-')[1]), int(line.split('-')[2])
            node0 = field.Graph.nodes[f"line_0-{field_idx}-{line_idx}"]['coord']
            node1 = field.Graph.nodes[f"line_1-{field_idx}-{line_idx}"]['coord']
            node0, node1 = find_point(ori_field.Graph, node0), find_point(ori_field.Graph, node1)
            if node0:
                field_idx, line_idx = int(node0.split('-')[1]), int(node0.split('-')[2])
            else:
                field_idx, line_idx = int(node1.split('-')[1]), int(node1.split('-')[2])
            
            a[0] = f'line-{field_idx}-{line_idx}'

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

starts = ['bound-3-2', 'bound-3-0', 'bound-2-0', 'bound-1-0']
ends = ['bound-3-0', 'bound-2-0', 'bound-1-0', 'bound-3-2']
field_ends = ['bound-3-2' for _ in range(5)]
field_ids = [3, 2, 0, 1]
figs = []
s, t, c = 0, 0, 0
status = []

field = load_dict(os.path.join(data, 'field.pkl'))
ax = field.render(working_lines=False, show=False, start=False, end=False)
ax.plot(field.Graph.nodes[starts[0]]['coord'][0], 
                field.Graph.nodes[starts[0]]['coord'][1], 
                '*y', 
                markersize=20)
ax.set_title(f'$s_P$={np.round(s, 2)}m, $t_P$={np.round(t, 2)}s, $c_P$={np.round(c, 2)}L', fontsize=20)

for idx in range(9):
    T = []
    finished = []
    if idx == 0:
        field = load_dict(os.path.join(data, 'field.pkl'))
        field.starts = [starts[0] for _ in range(5)]
        field.end = [ends[-1] for _ in range(5)]
        T += arrangement([0, 1], [field_ids[idx]], [], ends[idx], car_cfg, field, ac, [])
        field_ends[0] = ends[idx]
        field_ends[1] = ends[idx]        

    else:
        cidx = min(idx, 3)
        field = load_dict(os.path.join(data, 'field.pkl'))
        field.starts = [starts[0] for _ in range(5)]
        field.end = [ends[-1] for _ in range(5)]
        T += arrangement([0, 1], field_ids[:cidx+1], status, ends[cidx], car_cfg, field, ac, unfinished)
        field_ends[0] = ends[cidx]
        field_ends[1] = ends[cidx]
        if len(T[0]) == 0:
            finished.append(0)
        if len(T[1]) == 0:
            finished.append(1)

    if idx > 0:
        cidx = min(idx-1, 3)
        fidx = [0, 1, 2, 2, 3, 4, 4, 4, 4]
        field = load_dict(os.path.join(data, 'field.pkl'))
        field.starts = [starts[0] for _ in range(5)]
        field.end = [ends[-1] for _ in range(5)]
        T += arrangement([2], field_ids[:fidx[idx]], status, ends[cidx], car_cfg, field, ac, unfinished)
        field_ends[2] = ends[cidx]
        if len(T[2]) == 0:
            finished.append(2)
    else:
        T += [[]]
        field_ends[2] = ends[-1]        

    if idx > 1:
        cidx = min(idx-2, 3)
        field = load_dict(os.path.join(data, 'field.pkl'))
        field.starts = [starts[0] for _ in range(5)]
        field.end = [ends[-1] for _ in range(5)]
        T += arrangement([3, 4], field_ids[:cidx+1], status, ends[cidx], car_cfg, field, ac, unfinished)
        field_ends[3] = ends[cidx]
        field_ends[4] = ends[cidx]
        if len(T[3]) == 0:
            finished.append(3)
        if len(T[4]) == 0:
            finished.append(4)
    else:
        T += [[], []]
        field_ends[3] = ends[-1]
        field_ends[4] = ends[-1]
    
    field = load_dict(os.path.join(data, 'field.pkl'))
    if idx > 0:
        field.starts = [starts[0] for _ in range(5)]
        field.ends = field_ends
        edit_fields(field, status, field_ends, veh_sel=[0, 1, 2, 3, 4], delete_lines=False)
    else:
        field.starts = [starts[0] for _ in range(5)]
        field.ends = field_ends
        field.num_veh = len(field.starts)
        field.num_endpoints = 2*field.num_veh
        field.make_working_graph()
    for veh in T:
        for a in veh:
            a[0] = field.working_line_list.index(a[0])
    
    simulator = arrangeSimulator(field, car_cfg)
    simulator.init_simulation(T)
    if idx < 8:
        simulator.simulator.dynamic_terminate = True
    else:
        simulator.simulator.dynamic_terminate = False
    if idx == 0:    
        simulator.simulator.veh_terminate_idx = [0]
    elif idx == 1:
        simulator.simulator.veh_terminate_idx = [1]
    elif idx == 2:
        simulator.simulator.veh_terminate_idx = [1]
    elif idx == 3:
        simulator.simulator.veh_terminate_idx = [2]
    elif idx == 4:
        simulator.simulator.veh_terminate_idx = [4]
    else:
        simulator.simulator.veh_terminate_idx = [3]
    # t0, c0, s0, _, _, _, figs0 = simulator.simulate(ax, False, False, False, False)
    t0, c0, s0, _, _, _, figs0 = simulator.simulate(ax, True, True, True, False)
    s += s0
    t += t0
    c += c0
    figs += figs0
    unfinished = [[] for _ in range(5)]
    for i, stat in enumerate(simulator.simulator.car_status):
        unfinished[i] += T[i][len(stat['traveled'])+1:]
        for a in unfinished[i]:
            a[0] = field.working_line_list[a[0]]

    if idx > 0:
        for i in range(5):
            simulator.simulator.car_status[i]['traveled'] += status[i]['traveled']
    
    status = simulator.simulator.car_status
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

case = 'field_fusion'
save_dir = os.path.join(data, case + ".mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(save_dir, fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
# map(videoWriter.write, figs)
for fig in figs:
    videoWriter.write(fig)
videoWriter.release() 
    
print(save_dir)