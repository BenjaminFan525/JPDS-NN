import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Tuple, Union, List
import torch_geometric.nn as gnn
from torch_geometric.utils import unbatch, from_networkx, to_dense_adj, scatter
import model.common as nn_common
from torch_geometric.data import Data, Batch
from copy import deepcopy
import numpy as np

class GNNWithEdge(nn.Module):
    def __init__(self, node_dim, edge_dim, embedding_layer=1, embed_dim=64, nhead=4, 
                 activation=F.relu, layer_num = 2, dropout: float = 0.1,
                 whole_batch_edge_processing=False, edge_norm: bool = True, 
                 device='cpu', dtype=torch.float32) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(activation, str):
            activation = nn_common.activations[activation]
        super().__init__()
        self.activation = activation
        self.embed_dim = embed_dim

        self.node_embedding = nn_common.Embedding_layer(node_dim, self.embed_dim, embedding_layer,
                                                        activation=self.activation, **self.factory_kwargs)
        self.edge_embedding = nn_common.Embedding_layer(edge_dim, self.embed_dim, embedding_layer,
                                                        activation=self.activation, **self.factory_kwargs)
        # edge feature
        self.whole_batch_edge_processing = whole_batch_edge_processing
        self.edge_attn = nn.MultiheadAttention(self.embed_dim, nhead, **self.factory_kwargs)
        self.edge_linear1 = nn.Linear(self.embed_dim, self.embed_dim, **self.factory_kwargs)
        self.edge_linear2 = nn.Linear(self.embed_dim, self.embed_dim, **self.factory_kwargs)
        self.edge_norm = edge_norm
        if self.edge_norm:
            self.edge_norm1 = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)
            self.edge_norm2 = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)
        self.edge_dropout1 = nn.Dropout(dropout)
        self.edge_dropout2 = nn.Dropout(dropout)

        # node feature (main GNN part)
        self.attn_head_dim = self.embed_dim // nhead
        assert self.attn_head_dim * nhead == self.embed_dim, "embed_dim must be divisible by num_heads"
        GNN_layer = gnn.TransformerConv(self.embed_dim, self.attn_head_dim, nhead, 
                                    edge_dim=self.embed_dim, dropout=dropout, 
                                    **self.factory_kwargs).to(self.factory_kwargs['device'])
        self.mods = nn.ModuleList([deepcopy(GNN_layer) for _ in range(layer_num)])
        norm_layer = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)
        self.norms = nn.ModuleList([deepcopy(norm_layer) for _ in range(layer_num)])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def edge_feature(self, e: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.edge_norm:
            e = self.edge_norm1(e + self.edge_attn_block(e, batch))
            e = self.edge_norm2(e + self.edge_ff(e))
        else:
            e = e + self.edge_attn_block(e, batch)
            e = e + self.edge_ff(e)
        return e

    def edge_ff(self, e: torch.Tensor) -> torch.Tensor:
        return self.edge_dropout2(self.edge_linear2(self.activation(self.edge_linear1(e))))

    def edge_attn_block(self, e: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            return self.edge_dropout1(self.edge_attn(e, e, e, need_weights=False)[0])
        
        if self.whole_batch_edge_processing:
            bsz = batch[-1] + 1
            attn_mask = torch.ones((len(batch), len(batch)), dtype=torch.bool, device=e.device)
            start = 0
            for idx in range(bsz):
                num = (batch == idx).sum().item()
                end = start + num
                attn_mask[start:end, start:end] = torch.zeros((num, num), dtype=torch.bool, device=e.device)
                start = end
            return self.edge_dropout1(self.edge_attn(e, e, e, attn_mask=attn_mask, need_weights=False)[0])
        else:
            attn_output = [self.edge_attn(x, x, x, need_weights=False)[0] for x in unbatch(e, batch)]
            return self.edge_dropout1(torch.concat(attn_output))
        
    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if data.batch is None:
            edge_to_graph = None
        else:
            edge_to_graph = data.batch[edge_index[0]]

        x = self.node_embedding(x)
        e = self.edge_embedding(edge_attr)

        e = self.edge_feature(e, edge_to_graph)
        # assert not torch.isnan(x).any()
        # assert not torch.isnan(e).any()

        for mod, norm in zip(self.mods, self.norms):
            x = norm(x + mod(x, edge_index, e))

        return x

class ClusterGNN(nn.Module):
    default_common = {
        'embed_dim': 64,
        'activation': 'relu',
    }

    default_gnn = {
        'node_dim': 3, 
        'edge_dim': 4, 
        'embedding_layer': 1,
        'nhead': 4, 
        'layer_num': 2, 
        'dropout': 0.1,
        'whole_batch_edge_processing': False, 
        'edge_norm': True,
    }

    default_ff = {
        'sigmoid_tau': 2,
        'output_dim': [32, 16],
    }

    def __init__(self, common_cfg=None, gnn_cfg=None, ff_cfg=None, device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.common = deepcopy(self.default_common)
        self.gnn_cfg = deepcopy(self.default_gnn)
        self.ff_cfg = deepcopy(self.default_ff)

        if common_cfg is not None:
            self.common.update(common_cfg)
        if gnn_cfg is not None:
            self.gnn_cfg.update(gnn_cfg)
        if ff_cfg is not None:
            self.ff_cfg.update(ff_cfg)

        self.cfg = {
            'gnn': deepcopy(self.gnn_cfg),
            'ff': deepcopy(self.ff_cfg)
        }
        self.cfg['gnn'].update(self.common)
        self.cfg['ff'].update(self.common)
        self.gnn = GNNWithEdge(**self.common, **self.gnn_cfg, **self.factory_kwargs)
        
        if isinstance(self.common['activation'], str):
            self.common['activation'] = nn_common.activations[self.common['activation']]
        self.activation = self.common['activation']
        linear_layers = [nn.Linear(self.common['embed_dim'], self.ff_cfg['output_dim'][0], **self.factory_kwargs)]
        for dim1, dim2 in zip(self.ff_cfg['output_dim'][:-1], self.ff_cfg['output_dim'][1:]):
            linear_layers.append(nn.Linear(dim1, dim2, **self.factory_kwargs))
        self.ff =nn.ModuleList(linear_layers)
        self.tau = self.ff_cfg['sigmoid_tau']
    
    def forward(self, data: Data):
        # if isinstance(data, list):
        #     x = [self.gnn(d) for d in data]
        #     x = torch.concat(x, dim=0)
        # else:
        x = self.gnn(data)
        # assert not torch.isnan(x).any()
        
        for mod in self.ff[:-1]:
            x = self.activation(mod(x))
        
        x = self.ff[-1](x)

        x = F.normalize(x, dim=-1)
        # assert not torch.isnan(x).any()

        if data.batch is None:
            return torch.sigmoid(self.tau * x @ x.t()) # normalize the cosine similarity to (0, 1)
        # #[torch.clip(0.5 * x @ x.t() + 0.5, 1e-5, 1 - 1e-5)]
        # batch = []
        # for idx, d in enumerate(data):
        #     batch += [idx] * d.x.shape[0]
        # batch = torch.Tensor(np.array(batch))
        attn_output = [torch.sigmoid(self.tau * p @ p.t()) for p in unbatch(x, data.batch)]
        #
        # [torch.clip(0.5 * p @ p.t() + 0.5, 1e-5, 1 - 1e-5) for p in unbatch(x, data.batch)]
        return attn_output

class GNNEncoder(nn.Module):
    default_common = {
        'embed_dim': 64,
        'activation': 'relu',
    }
    
    default_gnn = {
        'node_dim': 3, 
        'edge_dim': 4, 
        'embedding_layer': 1,
        'nhead': 4, 
        'layer_num': 2, 
        'dropout': 0.1,
        'whole_batch_edge_processing': False, 
        'edge_norm': True,
    }

    default_gff = {
        'layer': 2
    }

    def __init__(self, common_cfg=None, gnn_cfg=None, gff_cfg=None, frozen_gnn=False, 
                 device='cpu', dtype=torch.float32) -> None:
        
        super().__init__()
        self.common_cfg = deepcopy(self.default_common)
        self.gnn_cfg = deepcopy(self.default_gnn)
        self.gff_cfg = deepcopy(self.default_gff)
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        if common_cfg is not None:
            self.common_cfg.update(common_cfg)
        if gnn_cfg is not None:
            self.gnn_cfg.update(gnn_cfg)
        if gff_cfg is not None:
            self.gff_cfg.update(gff_cfg)

        if isinstance(self.common_cfg['activation'], str):
            self.activation = nn_common.activations[self.common_cfg['activation']]
        else:
            self.activation = self.common_cfg['activation']
        
        self.frozen_gnn = frozen_gnn
        if 'activation' in self.gnn_cfg:
            self.gnn_cfg.update({'embed_dim': self.common_cfg['embed_dim']})
        else:
            self.gnn_cfg.update(self.common_cfg)
        self.gnn = GNNWithEdge(**self.gnn_cfg, **self.factory_kwargs)
        self.global_embedding = nn_common.Embedding_layer(2 * self.common_cfg['embed_dim'], 
                                                          self.common_cfg['embed_dim'], 
                                                          self.gff_cfg['layer'], activation=self.activation, 
                                                          **self.factory_kwargs)
    
    def freeze_gnn(self):
        self.frozen_gnn = True

    def unfreeze_gnn(self):
        self.frozen_gnn = False

    def forward(self, data: Data):
        if self.frozen_gnn:
            with torch.no_grad():
                x = self.gnn(data)
        else:
            x = self.gnn(data)
        if data.batch is None:
            g = torch.unsqueeze(torch.cat([torch.mean(x, dim=0), torch.max(x, dim=0)[0]]), 0)
            x = torch.unsqueeze(x, 0)
            key_padding_mask = torch.zeros((x.shape[0], x.shape[1]), device=x.device).bool()
        else:
            g = torch.cat([
                scatter(x, data.batch, dim=0, reduce='mean'),
                scatter(x, data.batch, dim=0, reduce='max')
            ], dim=-1)
            x = unbatch(x, data.batch)
            key_padding_mask = [torch.zeros(len(d), device=d.device) for d in x]
            x = pad_sequence(x, batch_first=True, padding_value=0.)
            key_padding_mask = pad_sequence(key_padding_mask, batch_first=True, padding_value=1).bool()
        
        return self.global_embedding(g), x, key_padding_mask

if __name__ == '__main__':
    from env import multiField
    from torch_geometric.loader import DataLoader, DataListLoader
    import time
    from functools import partial

    batch_size = 4
    data_list = []
    target_list = []

    for _ in range(batch_size):
        field = multiField([1, 1], width = (1000, 1000), working_width=24) 

        pygdata = from_networkx(field.working_graph, 
                                group_node_attrs=['embed'],
                                group_edge_attrs=['embed'])
        target = torch.zeros((field.num_nodes, field.num_nodes))
        target[0, 0] = 1.
        start = 1
        for f in field.fields:
            target[start:start + f.num_working_lines, start:start + f.num_working_lines] = torch.ones((f.num_working_lines, f.num_working_lines))
            start += f.num_working_lines
        target_list.append(target)
        # pygdata.y = target
        # pygdata.x = pygdata.x.type(torch.float16)
        # pygdata.edge_attr = pygdata.edge_attr.type(torch.float16)
        data_list.append(pygdata)
    # print(pygdata)
    loader = DataLoader(data_list, batch_size=batch_size)
    batch = next(iter(loader))

    device = 'cuda:0'
    model1 = ClusterGNN(device=device)
    
    # model1 = model1.type(torch.float16)
    model2 = ClusterGNN(gnn_cfg={
        'node_dim': 3, 
        'edge_dim': 4, 
        'nhead': 4, 
        'layer_num': 2, 
        'dropout': 0.1,
        'whole_batch_edge_processing': True, 
        'edge_norm': True,
    }, device=device)
    # model1.eval()

    hooks = {}

    def forward_hook(module, data_input, data_output, fmap_block, input_block):
        fmap_block.append(data_output)
        input_block.append(data_input)

    # 注册Hook
    for name, mod in model1.named_modules():
        hooks.update({name: ([], [])})
        hook_fn = partial(forward_hook, fmap_block=hooks[name][0], input_block=hooks[name][1])
        mod.register_forward_hook(hook_fn)

    batch.to(device)
    

    for data in data_list:
        data.to(device)
    # with torch.no_grad():
    tic = time.time()
    output = model1(batch)
    print(time.time() - tic)
    tic = time.time()
    output1 = model1(batch)
    print(time.time() - tic)
    tic = time.time()
    output2 = [model1(data) for data in data_list]
    print(time.time() - tic)
    # print(output1 == output2)
    # tic = time.time()
    # output = model2(batch)
    # print(time.time() - tic)
    criterion = nn.BCELoss()
    loss = torch.zeros(batch_size)
    for idx in range(batch_size):
        loss[idx] = criterion(output[idx], target_list[idx].to(device))
        # loss = torch.mean(torch.Tensor([criterion(out, tar.to(device)) for out, tar in zip(output, target_list)]))
    loss = torch.mean(loss)
    loss.backward()

    enc = GNNEncoder(device=device)
    g, x, key_padding_mask = enc(batch)
    pass



