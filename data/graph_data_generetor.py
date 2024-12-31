from env import multiField
import torch
from torch_geometric.utils import unbatch, from_networkx
import argparse
import os
from utils import save_dict, load_dict
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_num', type=int, default=100)
    parser.add_argument('--field_num', type=int, default=1)
    parser.add_argument('--veh_num', type=int, default=1)
    parser.add_argument('--random_veh', action='store_true', default=False)
    # parser.add_argument('--field_edge', type=float, default=)
    parser.add_argument('--save_dir', type=str, default='/home/fanyx/mdvrp/data/Gdataset')
    # parser.add_argument('--prefix', type=str, default='0')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    args.save_dir = os.path.join(args.save_dir, str(args.field_num))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if args.field_num == 1:
        width_range = (300, 800)
        splits = [0, 0]
        field_type = '#'
    elif args.field_num == 2:
        width_range = (300, 800)
        splits = [1, 0]
        field_type = '#'
    elif args.field_num == 3:
        width_range = (400, 800)
        splits = [0, 0]
        field_type = 'T'
    elif args.field_num == 4:
        width_range = (500, 900)
        splits = [1, 1]
        field_type = '#'
    elif args.field_num == 6:
        width_range = (600, 700)
        splits = [2, 1]
        field_type = '#'

    total_node = 0
    for idx in tqdm(range(args.data_num)):
        width = (width_range[1] - width_range[0]) * np.random.random() + width_range[0]
        while True:
            if not args.random_veh:
                field = multiField(splits, type=field_type, width = (width, width), num_starts=args.veh_num, num_ends=args.veh_num) 
            else:
                random_num = random.randint(1, 2*args.field_num)
                field = multiField(splits, type=field_type, width = (width, width), num_starts=random_num, num_ends=random_num) 
            node_num = [f.working_graph.number_of_nodes() for f in field.fields]
            if 0 in node_num:
                continue
            else:
                break
            
        ax = plt.subplot(1, 1, 1)
        field.render(ax, show=False)

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
        num_starts, num_ends = len(field.starts), len(field.ends)
        target[:num_starts, :num_starts] = torch.ones((num_starts, num_starts))
        target[num_starts:num_starts+num_ends, num_starts:num_starts+num_ends] = torch.ones((num_ends, num_ends))
        start = num_starts + num_ends
        for f in field.fields:
            target[start:start + f.num_working_lines, start:start + f.num_working_lines] = torch.ones((f.num_working_lines, f.num_working_lines))
            start += f.num_working_lines

        save_dict(pygdata, os.path.join(name, 'pygdata'))
        save_dict(target, os.path.join(name, 'cluster_target'))

        total_node += field.num_nodes
    
    with open(os.path.join(args.save_dir, 'dataset_config.txt'), 'a') as f:
        f.write(f'Total nodes: {total_node}')
