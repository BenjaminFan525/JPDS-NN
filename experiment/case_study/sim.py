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
import ia.algo.arrangement.models.ac as ia_model
import ia.utils as ia_util
import ia.env.simulator as ia_env
from matplotlib.animation import FuncAnimation

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

data = "/home/fanyx/mdvrp/experiment/case_study/4_4/0_39_24"
field_ia = load_dict(os.path.join(data, 'field.pkl'))


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

checkpoint_ia = "/home/fanyx/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-04__12-53_t/best_model39.pt"
checkpoint = "/home/fanyx/mdvrp/result/training_rl/ppo/2024-12-27__12-11/best_model31.pt"

car_tensor = torch.tensor(np.array([[cur_car['vw'], cur_car['vv'],
                cur_car['cw'], cur_car['cv'],
                cur_car['tt']] for cur_car in car_cfg])).float()

field = multiField(num_splits=[1, 1], 
                       starts=['bound-2-1', 
                               'bound-2-1', 
                               'bound-2-1', 
                               'bound-2-1', 
                               'bound-2-1'],
                       ends=['bound-2-1',
                             'bound-2-1',
                             'bound-2-1',
                             'bound-2-1',
                             'bound-2-1']
                       )

field.from_ia(field_ia)
# field_list = [0, 3]
# field.make_working_graph(field_list)
# field.make_working_graph()

ac = load_model(checkpoint)
ac.sequential_sel = True

pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
info = {'veh_key_padding_mask': torch.zeros((1, len(car_tensor))).bool(), 'num_veh': torch.tensor([[car_tensor.shape[0]]])}

with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, )
T = decode(seq_enc[0])

field_ia = load_dict(os.path.join(data, 'field.pkl'))
ac_ia = load_model(checkpoint_ia, model_type='ia')
pygdata = from_networkx(field_ia.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
info = {'veh_key_padding_mask': torch.zeros((1, len(car_tensor))).bool(), 'num_veh': torch.tensor([[car_tensor.shape[0]]])}
with torch.no_grad():
    seq_enc, _, _, _, _, _, _ = ac_ia(data_t, info, deterministic=True, criticize=False, )

T_ia = ia_util.decode(seq_enc[0])

for idx, T in enumerate([T, T_ia]):
    print(f"Simulator is initialing...")
    simulator = arrangeSimulator(field, car_cfg)
    simulator.init_simulation(T, debug=True)
    simulator.render_arrange()

    a = np.array([[cfg['vw'], cfg['vv']] for cfg in simulator.car_cfg])

    t, c, s, car_time, car_dis, figs = simulator.simulate(True, True, False)

    s, t, c = fit(field.D_matrix, 
                    field.ori, 
                    field.des,
                    car_cfg, 
                    field.line_length,
                    T, 
                    tS_t=False,
                    type='all')
    plt.title(f'$s_P$={np.round(s, 2)}m, $t_P$={np.round(t, 2)}s, $c_P$={np.round(c, 2)}L', fontsize=20)
    print(f"Simulation result: s={np.round(s, 2)}m, t={np.round(t, 2)}s, c={np.round(c, 2)}L")
    

plt.show()


# chosen_idx = [[line[0] for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]
# chosen_entry = [[line[1] for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]

# if figs is not None:
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     videoWriter = cv2.VideoWriter(os.path.join(data, algo + ".mp4"), fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
#     # map(videoWriter.write, figs)
#     for fig in figs:
#         videoWriter.write(fig)
#     videoWriter.release() 
# else:
#     print('error')
    
# print(os.path.join(data, algo + ".mp4"))