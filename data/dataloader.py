import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.data.on_disk_dataset import OnDiskDataset
from torch_geometric.loader.dataloader import Collater
from typing import Any, List, Optional, Sequence, Union


class ClusterGNNDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.collator = Collater(dataset, follow_batch, exclude_keys)

        if isinstance(dataset, OnDiskDataset):
            dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs,
        )
    
    def collate_fn(self, raw_data: List[List]):
        graph_data = [data[0] for data in raw_data]
        target = [data[1] for data in raw_data]
        field = [data[2] for data in raw_data]
        graph_data = self.collator.collate_fn(graph_data)
        return graph_data, target, field
    

class RLDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.collator = Collater(dataset, follow_batch, exclude_keys)

        if isinstance(dataset, OnDiskDataset):
            dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs,
        )
    
    def collate_fn(self, raw_data: List[List]):
        graph_data = [data[0] for data in raw_data]
        car_tensor = [data[1] for data in raw_data]
        field = [data[2] for data in raw_data]
        car_cfg = [data[3] for data in raw_data]
        cost_limits = raw_data[0][4]
        for key in cost_limits.keys():
            cost_limits[key] = torch.Tensor(np.array([data[4][key] for data in raw_data])).float().view(-1, 1)

        bsz_data = {'graph': self.collator.collate_fn(graph_data), 
                    'vehicles': pad_sequence(car_tensor, batch_first=True).float()}
        
        info = {'veh_key_padding_mask': [], 'num_veh': []}
        info['veh_key_padding_mask'] = [torch.zeros(len(x)) for x in car_tensor]
        info['veh_key_padding_mask'] = pad_sequence(info['veh_key_padding_mask'], batch_first=True, padding_value=1).bool()
        
        info['num_veh'] = [x.shape[0] for x in car_tensor]
        info['num_veh'] = torch.tensor(np.array(info['num_veh'])).view(-1, 1)
        
        return bsz_data, info, field, car_cfg, cost_limits


class RLTestDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.collator = Collater(dataset, follow_batch, exclude_keys)

        if isinstance(dataset, OnDiskDataset):
            dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs,
        )
    
    def collate_fn(self, raw_data: List[List]):
        graph_data = [data[0] for data in raw_data]
        graph_data_ia = [data[1] for data in raw_data]
        car_tensor = [data[2] for data in raw_data]
        field = [data[3] for data in raw_data]
        field_ia = [data[4] for data in raw_data]
        car_cfg = [data[5] for data in raw_data]
        cost_limits = None

        bsz_data = {'graph': self.collator.collate_fn(graph_data),
                    'vehicles': pad_sequence(car_tensor, batch_first=True).float()}
        
        bsz_data_ia = {'graph': self.collator.collate_fn(graph_data_ia),
                    'vehicles': pad_sequence(car_tensor, batch_first=True).float()}
        
        info = {'veh_key_padding_mask': [], 'num_veh': []}
        info['veh_key_padding_mask'] = [torch.zeros(len(x)) for x in car_tensor]
        info['veh_key_padding_mask'] = pad_sequence(info['veh_key_padding_mask'], batch_first=True, padding_value=1).bool()
        
        info['num_veh'] = [x.shape[0] for x in car_tensor]
        info['num_veh'] = torch.tensor(np.array(info['num_veh'])).view(-1, 1)
        
        return bsz_data, bsz_data_ia, info, field, field_ia, car_cfg, cost_limits