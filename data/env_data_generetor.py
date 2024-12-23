from mdvrp.env import multiField
import torch
from torch_geometric.utils import unbatch, from_networkx
import argparse
import os
from mdvrp.utils import save_dict, load_dict
import numpy as np
import random
import matplotlib.pyplot as plt

config = {
    '12': {
        'xs': { # everage 20
            '1': (380, 380),
            '2': (360, 380),
            '3': (340, 340),
            '4': (320, 320),
            '6': (320, 320),
        },
        's': { # everage 30
            '1': (500, 650),
            '2': (450, 600),
            '3': (400, 500),
            '4': (400, 500),
            '6': (400, 500),
        },
        'm': { # everage 40
            '1': (700, 800),
            '2': (600, 700),
            '3': (500, 700),
            '4': (500, 550),
            '6': (550, 550),
            '10': (800, 900),
        },
        'L': {
            '8': (1150, 1150),
            '10': (1100, 1150),
            '12': (925, 925),
        }
    },
    '24': {
        'xs': { # everage 20
            '1': (700, 750),
            '2': (700, 750),
            '3': (650, 670),
            '4': (550, 600),
            '6': (600, 600),
        },
        's': {  # everage 30
            '1': (900, 1200),
            '2': (900, 1100),
            '3': (800, 1000),
            '4': (700, 900),
            '6': (700, 800),
        },
        'm': { # everage 40
            '1': (1300, 1500),
            '2': (1100, 1400),
            '3': (1000, 1300),
            '4': (900, 1100),
            '6': (900, 950),
            '10': (1500, 1500)
        },
        'L': {
            '8': (1950, 2000),
            '10': (2100, 2100),
            '12': (1925, 1925),
        }
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_num', type=int, default='1')
    parser.add_argument('--field_num', type=int, default=4)
    parser.add_argument('--veh_num', type=int, default=4)
    # parser.add_argument('--field_edge', type=float, default=)
    parser.add_argument('--task_size', type=str, default='m')
    parser.add_argument('--save_dir', type=str, default='MAdata')
    # parser.add_argument('--prefix', type=str, default='0')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    args.save_dir = os.path.join(args.save_dir, str(args.field_num) + '_' + str(args.veh_num))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.field_num == 1:
        splits = [0, 0]
        field_type = '#'
    elif args.field_num == 2:
        splits = [1, 0]
        field_type = '#'
    elif args.field_num == 3:
        splits = [0, 0]
        field_type = 'T'
    elif args.field_num == 4:
        splits = [1, 1]
        field_type = '#'
    elif args.field_num == 6:
        splits = [2, 1]
        field_type = '#'
    elif args.field_num == 8:
        splits = [3, 1]
        field_type = '#'
    elif args.field_num == 10:
        splits = [5, 1]
        field_type = '#'
    elif args.field_num == 12:
        splits = [3, 2]
        field_type = '#'

    total_node = 0
    for idx in range(args.data_num):
        working_width = random.sample([12, 24], 1)[0]
        width_range = config[str(working_width)][args.task_size][str(args.field_num)]
        width = (width_range[1] - width_range[0]) * np.random.random() + width_range[0]
        node_num = 0
        while node_num <= 15:
            field = multiField(splits, type=field_type, width = (width, width), working_width=working_width) 
            node_num = field.num_nodes
            
        ax = plt.subplot(1, 1, 1)
        field.render(ax, show=False, entry_point=True, boundary=True, line_colors='k')

        name = f'{idx}_{field.num_nodes}_{field.working_width}'
        name = os.path.join(args.save_dir, name)
        if not os.path.exists(name):
            os.mkdir(name)

        save_dict(field, os.path.join(name, 'field'))
        plt.savefig(os.path.join(name, 'render.png'))
        ax.clear()

        pygdata = from_networkx(field.working_graph, 
                                group_node_attrs=['embed'],
                                group_edge_attrs=['edge_embed'])
        
        target = torch.zeros((field.num_nodes, field.num_nodes))
        target[0, 0] = 1.
        start = 1
        for f in field.fields:
            target[start:start + f.num_working_lines, start:start + f.num_working_lines] = torch.ones((f.num_working_lines, f.num_working_lines))
            start += f.num_working_lines
        
        save_dict(pygdata, os.path.join(name, 'pygdata'))
        save_dict(target, os.path.join(name, 'cluster_target'))

        car_cfg = []
        car_cfg_v = []
        
        for idx in range(args.veh_num):
            cur_car = {'vw': np.random.uniform(1, 3.3, 1).item(), 
                    'vv': 0, 
                    'cw': 0, 
                    'cv': np.random.uniform(0.005, 0.008, 1).item(),
                    'tt': 0, 
                    'min_R': working_width / 2 + np.random.uniform(-1, 1, 1).item()}
            cur_car['vv'] = np.random.uniform(max(2, cur_car['vw']), 25/3.6, 1).item()
            cur_car['cw'] = np.random.uniform(max(0.007, cur_car['cv']), 0.01, 1).item()
            car_cfg.append(cur_car)
            car_cfg_v.append([cur_car['vw'], cur_car['vv'],
                              cur_car['cw'], cur_car['cv'],
                              cur_car['tt']])
        car_cfg_v = torch.tensor(np.array(car_cfg_v))

        save_dict(car_cfg, os.path.join(name, 'car_cfg'))
        save_dict(car_cfg_v, os.path.join(name, 'car_tensor'))

        total_node += field.num_nodes
    
    with open(os.path.join(args.save_dir, 'dataset_config.txt'), 'a') as f:
        f.write(f'Total nodes: {total_node}')
