from random import shuffle
from torch.utils import data
from utils.common import load_dict
import os
import json


class ClusterGNNDataset(data.Dataset):
    def __init__(self, data: list) -> None:
        super().__init__()
        self.data = data
        self.use_field = False

    def __getitem__(self, index):
        return [load_dict(os.path.join(self.data[index], 'pygdata.pkl')),
                load_dict(os.path.join(self.data[index], 'cluster_target.pkl')),
                load_dict(os.path.join(self.data[index], 'field.pkl'))] if self.use_field else \
               [load_dict(os.path.join(self.data[index], 'pygdata.pkl')),
                load_dict(os.path.join(self.data[index], 'cluster_target.pkl')),
                None]
    
    def __len__(self):
        return len(self.data)


class MaDataset(data.Dataset):
    def __init__(self, data: list) -> None:
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return load_dict(self.data[index])
    
    def __len__(self):
        return len(self.data)


class RLDataset(data.Dataset):
    def __init__(self, data: list, reciprocal: bool=False, cost_dict = None, use_field: bool=False) -> None:
        super().__init__()
        self.data = data
        self.reciprocal = reciprocal
        self.cost_dict = cost_dict
        self.use_field = use_field

    def __getitem__(self, index):
        d = [load_dict(os.path.join(self.data[index], 'pygdata.pkl')),
                load_dict(os.path.join(self.data[index], 'car_tensor.pkl')),
                load_dict(os.path.join(self.data[index], 'field.pkl')),
                load_dict(os.path.join(self.data[index], 'car_cfg.pkl'))] if self.use_field else \
            [load_dict(os.path.join(self.data[index], 'pygdata.pkl')),
                load_dict(os.path.join(self.data[index], 'car_tensor.pkl')),
                load_dict(os.path.join(self.data[index], 'field_matrices.pkl')),
                load_dict(os.path.join(self.data[index], 'car_cfg.pkl'))]
        
        cost_limits = {}
        if self.cost_dict is not None:
            for key, value in self.cost_dict.items():
                with open(os.path.join(self.data[index], value)) as f:
                    cfg = json.load(f)
                    cost_limits.update({
                        key: cfg[key]
                    })
        d.append(cost_limits)
        
        if self.reciprocal:
            d[1][:, 0] = 1 / d[1][:, 0]
            d[1][:, 1] = 1 / d[1][:, 1]
            d[1][:, 2] = d[1][:, 2] * d[1][:, 0]
            d[1][:, 3] = d[1][:, 3] * d[1][:, 1]
        return d
    
    def __len__(self):
        return len(self.data)


class RLTestDataset(data.Dataset):
    def __init__(self, data: list, use_field: bool=False) -> None:
        super().__init__()
        self.data = data
        self.use_field = use_field

    def __getitem__(self, index):
        return [load_dict(os.path.join(self.data[index], 'pygdata.pkl')),
                load_dict(os.path.join(self.data[index], 'pygdata_ia.pkl')),
                load_dict(os.path.join(self.data[index], 'car_tensor.pkl')),
                load_dict(os.path.join(self.data[index], 'field.pkl')),
                load_dict(os.path.join(self.data[index], 'field_ia.pkl')),
                load_dict(os.path.join(self.data[index], 'car_cfg.pkl')),
                ] if self.use_field else \
                [load_dict(os.path.join(self.data[index], 'pygdata.pkl')),
                load_dict(os.path.join(self.data[index], 'pygdata_ia.pkl')),
                load_dict(os.path.join(self.data[index], 'car_tensor.pkl')),
                load_dict(os.path.join(self.data[index], 'field_matrices.pkl')),
                load_dict(os.path.join(self.data[index], 'field_matrices_ia.pkl')),
                load_dict(os.path.join(self.data[index], 'car_cfg.pkl')),
                ]
    
    def __len__(self):
        return len(self.data)