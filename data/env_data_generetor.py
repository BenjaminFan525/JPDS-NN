from env import multiField
import torch
from torch_geometric.utils import unbatch, from_networkx
import argparse
import os
from utils import save_dict, load_dict
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import ia.env.multi_field

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
            '1': (700, 800),
            '2': (600, 700),
            '3': (500, 700),
            '4': (700, 700),
            '5': (700, 800),
            '6': (800, 900),
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
            '1': (1300, 1500),
            '2': (1100, 1400),
            '3': (1000, 1300),
            '4': (1300, 1300),
            '5': (1300, 1500),
            '6': (1400, 1600),
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


def data_gen(data_num, field_num, veh_num, task_size, save_dir, save_ia, single_end):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_dir = os.path.join(save_dir, str(field_num) + '_' + str(veh_num))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if field_num == 1:
        splits = [0, 0]
        field_type = '#'
    elif field_num == 2:
        splits = [1, 0]
        field_type = '#'
    elif field_num == 3:
        splits = [1, 1]
        field_type = '#'
    elif field_num == 4:
        splits = [1, 1]
        field_type = '#'
    elif field_num == 5:
        splits = [2, 1]
        field_type = '#'
    elif field_num == 6:
        splits = [2, 1]
        field_type = '#'
    elif field_num == 8:
        splits = [3, 1]
        field_type = '#'
    elif field_num == 10:
        splits = [5, 1]
        field_type = '#'
    elif field_num == 12:
        splits = [3, 2]
        field_type = '#'

    total_node = 0
    for idx in tqdm(range(data_num)):
        working_width = random.sample([12, 24], 1)[0]
        width_range = config[str(working_width)][task_size][str(field_num)]
        width = (width_range[1] - width_range[0]) * np.random.random() + width_range[0]
        while True:
            if save_ia:
                field_ia = ia.env.multi_field.multiField(splits, type=field_type, width = (width, width), working_width=working_width) 
                field = multiField(splits, type=field_type, width = (width, width), working_width=working_width,
                                starts=[field_ia.home for _ in range(veh_num)],
                                ends=[field_ia.home for _ in range(veh_num)]) 
                field.from_ia(field_ia)
                if field_num == 3 or field_num == 5:
                    field_ia.merge_field([0, 1])
                    field.starts=[field_ia.home for _ in range(veh_num)]
                    field.ends=[field_ia.home for _ in range(veh_num)]
                    field.merge_field([0, 1])
            else:
                field = multiField(splits, type=field_type, width = (width, width), working_width=working_width,
                                num_starts=veh_num, num_ends=veh_num, single_end=single_end) 
                if field_num == 3 or field_num == 5:
                    field.merge_field([0, 1], num_starts=veh_num, num_ends=veh_num)
            
            line_nums = [f.num_working_lines for f in field.fields]
            if np.all(np.array(line_nums) > 3) == False or np.sum(line_nums) < 20:
                continue
            else:
                break
            
        ax = plt.subplot(1, 1, 1)
        field.render(ax, show=False, entry_point=True, boundary=True, line_colors='k')

        name = f'{idx}_{field.num_nodes}_{field.working_width}'
        name = os.path.join(save_dir, name)
        if not os.path.exists(name):
            os.mkdir(name)

        save_dict(field, os.path.join(name, 'field'))
        if save_ia:
            save_dict(field_ia, os.path.join(name, 'field_ia'))
        plt.savefig(os.path.join(name, 'render.png'))
        ax.clear()

        field_matrices = [field.D_matrix, field.ori, field.des, field.line_length]
        save_dict(field_matrices, os.path.join(name, 'field_matrices'))
        if save_ia:
            field_matrices_ia = [field_ia.D_matrix, field_ia.ori, field_ia.ori, field_ia.line_length]
            save_dict(field_matrices_ia, os.path.join(name, 'field_matrices_ia'))

        pygdata = from_networkx(field.working_graph, 
                                group_node_attrs=['embed'],
                                group_edge_attrs=['edge_embed'])
        save_dict(pygdata, os.path.join(name, 'pygdata'))
        if save_ia:
            pygdata_ia = from_networkx(field_ia.working_graph, 
                                    group_node_attrs=['embed'],
                                    group_edge_attrs=['edge_embed'])
            save_dict(pygdata_ia, os.path.join(name, 'pygdata_ia'))            

        car_cfg = []
        car_cfg_v = []
        
        for idx in range(veh_num):
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
    
    with open(os.path.join(save_dir, 'dataset_config.txt'), 'a') as f:
        f.write(f'Total nodes: {total_node}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_num', type=int, default=10)
    parser.add_argument('--field_num', nargs='+', type=int, default=[5, 2, 3])
    parser.add_argument('--veh_num', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--save_ia', action='store_true', default=False)
    parser.add_argument('--single_end', action='store_true', default=False)
    # parser.add_argument('--field_edge', type=float, default=)
    parser.add_argument('--task_size', type=str, default='s')
    parser.add_argument('--save_dir', type=str, default='/home/fanyx/mdvrp/data/Gdataset/Task_test_xxx')
    # parser.add_argument('--prefix', type=str, default='0')
    args = parser.parse_args()

    for f_num in args.field_num:
        for v_num in args.veh_num:
            data_gen(args.data_num, f_num, v_num, args.task_size, args.save_dir, args.save_ia, args.single_end)

    
