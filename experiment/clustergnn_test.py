from model import ClusterGNN
from data import ClusterGNNDataLoader, ClusterGNNDataset
from env.multi_field import multiField
from utils import fit, load_dict, decode
import numpy as np
from functools import partial
from torch_geometric.utils import unbatch, from_networkx
from torch_geometric.data import Batch
from torch_geometric.loader.dataloader import Collater
import matplotlib.pyplot as plt
import torch
import os
import json
import yaml
from utils import COLORS

gnn_checkpoint = '/home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs/2024-02-03__00-13_layer_4/best_model511.pt'
device = 'cpu'
model_cfg = '/home/xuht/Intelligent-Agriculture/ia/algo/arrangement/configs/clustergnn.yaml'
with open(model_cfg, 'r') as f:
    model_cfg = yaml.safe_load(f)
model = ClusterGNN(**model_cfg, device=device)
checkpoint = torch.load(gnn_checkpoint, map_location=device)
model.gnn.load_state_dict(checkpoint['gnn'])  
model.ff.load_state_dict(checkpoint['ff'])  
model.eval()

data = "/home/xuht/Intelligent-Agriculture/MAdata/4_4/0_39_24"
# data = "/home/xuht/Intelligent-Agriculture/MAdata/3_4/0_39_12"
field_ia = load_dict(os.path.join(data, 'field.pkl'))
car_cfg = [{'vw': 2.5, 'vv': 4.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
               {'vw': 3, 'vv': 5, 'cw': 0.007, 'cv': 0.005, 'tt': 0, 'min_R': 6},
               {'vw': 3, 'vv': 5.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
               {'vw': 2, 'vv': 5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
               {'vw': 3, 'vv': 6, 'cw': 0.01, 'cv': 0.008, 'tt': 0, 'min_R': 6},]
car_cfg_v = [[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for cur_car in car_cfg]
car_tensor = torch.tensor(np.array(car_cfg_v)).float()
field = multiField([1, 1], 
                   starts=['bound-2-1', 'bound-3-1', 'bound-0-1'],
                   ends=['bound-0-1', 'bound-1-1'])

# field = multiField([1, 1], homes=['bound-2-1', 'bound-2-1', 'bound-2-1',
#                                   'bound-2-1', 'bound-2-1'])
# field = multiField([1, 1], homes=['bound-2-1'])
field.from_ia(field_ia)

pygdata = from_networkx(field.working_graph, 
                        group_node_attrs=['embed'],
                        group_edge_attrs=['edge_embed'])
pygdata.to(device)

target = torch.zeros((field.num_nodes, field.num_nodes))
# target[0, 0] = 1.
# start = 1 
num_starts, num_ends = len(field.starts), len(field.ends)
target[:num_starts, :num_starts] = torch.ones((num_starts, num_starts))
target[num_starts:num_starts+num_ends, num_starts:num_starts+num_ends] = torch.ones((num_ends, num_ends))
start = num_starts + num_ends
for f in field.fields:
    target[start:start + f.num_working_lines, start:start + f.num_working_lines] = torch.ones((f.num_working_lines, f.num_working_lines))
    start += f.num_working_lines
target = [t.to(device) for t in target]

with torch.no_grad():
    output = model(pygdata)
feature = torch.pca_lowrank(output, 2)[0].cpu().numpy()
name = 'field.png'
save_dir = '/home/xuht/Intelligent-Agriculture/mdvrp/experiment'
path = os.path.join(save_dir, name)
ax = plt.subplot(1, 2, 1)
field.render(ax, show=False, line_colors = COLORS[:field.num_fields])
ax = plt.subplot(1, 2, 2)
for idx, line in enumerate(feature[:num_starts]):
    ax.plot(line[0], line[1], '*y')
for idx, line in enumerate(feature[num_starts : num_starts+num_ends]):
    ax.plot(line[0], line[1], 'vr')
for line, field_idx in zip(feature[num_starts+num_ends:], field.line_field_idx):
    ax.plot(line[0], line[1], 'o', color=COLORS[field_idx])
plt.savefig(path)
print(path)