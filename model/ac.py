import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Tuple, List
import torch.nn.functional as F
from copy import deepcopy
import random
from utils import decode
from model.common import FeatureBlock, activations, Embedding_layer, PositionalEncoding
from model.GNNs import GNNEncoder
from torch_geometric.data import Data
from torch_geometric.utils import unbatch, scatter
import os
import json

class MaPtrNet(Module):
    __constants__ = ['batch_first']

    def __init__(self, query_dim, embed_dim, bias=True, device=None, dtype=None, 
                 ptr_fn='attn', pnorm=False) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim

        self.bias = bias
        
        self.ptr_fn = ptr_fn
        self.pnorm = pnorm

        if self.ptr_fn == 'attn':
            self.q_proj_weight = nn.Linear(query_dim, embed_dim, bias, **self.factory_kwargs)
            self.k_proj_weight = nn.Linear(embed_dim, embed_dim, bias, **self.factory_kwargs)
        elif self.ptr_fn == 'ff':
            self.ff1 = nn.Linear(query_dim + embed_dim, embed_dim, bias, **self.factory_kwargs)
            self.ff2 = nn.Linear(embed_dim, 1, bias, **self.factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.ptr_fn == 'attn':
            nn.init.xavier_uniform_(self.q_proj_weight.weight)
            nn.init.xavier_uniform_(self.k_proj_weight.weight)

            if self.bias:
                nn.init.constant_(self.q_proj_weight.bias, 0.)
                nn.init.constant_(self.k_proj_weight.bias, 0.)

    def dist(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            tau: float = 1,
            ):
        key_padding_mask = torch.zeros((key.shape[0], key.shape[1] + 1), dtype=torch.bool, device=key.device) if key_padding_mask is None else key_padding_mask
        masked_ptr = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        
        if self.ptr_fn == 'attn':
            q = self.q_proj_weight(query)
            k = self.k_proj_weight(key)

            if self.pnorm:
                q = F.normalize(q, dim=-1, eps=1e-10)
                k = F.normalize(k, dim=-1, eps=1e-10)
                ptr = torch.bmm(q, k.transpose(2, 1))
            else:
                ptr = torch.tanh(torch.bmm(q, k.transpose(2, 1)) / torch.sqrt(torch.tensor(self.embed_dim)))
        elif self.ptr_fn == 'ff':
            ptr = torch.cat([key, query.repeat(1, key.shape[1], 1)], dim=-1)
            if self.pnorm:
                ptr = F.tanh(self.ff1(ptr))
                ptr = F.normalize(ptr, dim=-1, eps=1e-10)
                ptr = self.ff2(ptr)
            else:
                ptr = self.ff2(F.tanh(self.ff1(ptr)))
            ptr = torch.tanh(ptr.transpose(2, 1))
            
        # zero_attn_shape = (ptr.shape[0], 1, 1)
        # ptr = torch.cat([torch.zeros((zero_attn_shape), dtype=ptr.dtype, device=ptr.device), ptr], dim=-1)
        masked_ptr = masked_ptr.unsqueeze(1)
        key_padding_mask = key_padding_mask.unsqueeze(1)
        masked_ptr[~key_padding_mask] = ptr[~key_padding_mask]
        return F.softmax(masked_ptr / tau, dim=-1)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            deterministic: bool = False,
            tau = 1,
            idx = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        prob_v = self.dist(query, key, key_padding_mask)
        if idx is not None:
            index = idx.unsqueeze(1)
            prob = torch.gather(prob_v, dim=-1, index=index)
        elif deterministic:
            prob, index = prob_v.max(-1, keepdim=True)
        else:
            choice_soft = F.gumbel_softmax(torch.log(prob_v), tau=tau)
            index = choice_soft.max(-1, keepdim=True)[1]
            prob = torch.gather(prob_v, dim=-1, index=index)
        choice_hard = torch.zeros_like(prob_v, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        choice = choice_hard # - choice_soft.detach() + choice_soft

        return choice, index.squeeze(-1), prob.squeeze(-1)

class MaGNNEncoder(Module):
    default_common = {
        'embed_dim': 64,
        'activation': 'relu',
    }
    
    default_gnn_encoder = {
        'gnn_cfg': {
            'node_dim': 3, 
            'edge_dim': 4, 
            'embedding_layer': 1,
            'nhead': 4, 
            'layer_num': 2, 
            'dropout': 0.1,
            'whole_batch_edge_processing': False, 
            'edge_norm': True,
        },
        'gff_cfg': {
            'layer': 2
        }
    }

    def __init__(self, common_cfg=None, gnn_cfg=None, veh_dim=5, embedding_layer=1, enc_type='enc', 
                 layer_num=1, nhead=4, cross_attn_en = False, device=None, dtype=None) -> None:
        '''
        embedding_layer: number of layers of car embedding
        layer_num: number of layers of car FeatureBlock
        '''
        super().__init__()
        self.common_cfg = deepcopy(self.default_common)
        self.gnn_cfg = deepcopy(self.default_gnn_encoder)
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        if common_cfg is not None:
            self.common_cfg.update(common_cfg)
        if gnn_cfg is not None:
            self.gnn_cfg.update(gnn_cfg)

        if isinstance(self.common_cfg['activation'], str):
            self.activation = activations[self.common_cfg['activation']]
        else:
            self.activation = self.common_cfg['activation']

        self.embed_dim = self.common_cfg['embed_dim']
        self.veh_dim = veh_dim

        self.veh_embedding = Embedding_layer(veh_dim, self.embed_dim, embedding_layer, 
                                             activation=self.activation, **self.factory_kwargs)
        self.nodes_encoder = GNNEncoder(common_cfg=self.common_cfg, **self.gnn_cfg, **self.factory_kwargs)
             
        self.veh_attn = FeatureBlock(enc_type, layer_num, self.embed_dim, nhead, self.activation, **self.factory_kwargs)

        self.global_veh_ff = Embedding_layer(2 * self.embed_dim, self.embed_dim, 2, 
                                             activation=self.activation, **self.factory_kwargs)
        
        
        self.own_modules = [self.veh_embedding, self.nodes_encoder.global_embedding,
                            self.veh_attn, self.global_veh_ff]

        self.cross_attn_en = cross_attn_en
        if self.cross_attn_en:
            self.ff = nn.Linear(self.embed_dim, self.embed_dim, **self.factory_kwargs)
            self.cross_attn = nn.MultiheadAttention(self.embed_dim, nhead, batch_first=True, **self.factory_kwargs)
            self.own_modules += [self.ff, self.cross_attn]
        self.own_modules = nn.ModuleList(self.own_modules)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def freeze_gnn(self):
        self.nodes_encoder.freeze_gnn()
    
    def unfreeze_gnn(self):
        self.nodes_encoder.unfreeze_gnn()

    def forward(self, graph: Data, vehicles: torch.Tensor, veh_key_padding_mask: Optional[torch.Tensor] = None):
        # vehicle feature
        veh = self.veh_embedding(vehicles) 
        
        # readout of the graph, nodes' features, nodes' key padding mask
        globel_nodes, nodes, node_key_padding_mask = self.nodes_encoder(graph)

        globel_nodes = globel_nodes.unsqueeze(1)

        veh = self.veh_attn(veh, veh_key_padding_mask)
        
        if self.cross_attn_en:
            # veh_depot_attn = [self.cross_attn(veh, nodes[idx], nodes[idx], key_padding_mask=node_key_padding_mask[idx])[0][:,idx,:] for idx in range(self.depot_dim)]
            # veh = torch.cat(veh_depot_attn, dim=0).unsqueeze(0) + \
            #     veh + self.activation(self.ff(veh))
            
            # if self.end_en:
            #    endpoint_nodes = nodes[:, :2*self.veh_dim, :] 
            #    nodes = torch.cat([endpoint_nodes[:, 0::2, :]+endpoint_nodes[:, 1::2, :], nodes[:, 2*self.veh_dim:, :]], dim=1)
            #    node_key_padding_mask = node_key_padding_mask[:, self.veh_dim:]
            
            veh_depot_attn = []
            for idx in range(self.veh_dim):
                mask = node_key_padding_mask.clone()
                mask[:, :2*self.veh_dim] = True
                mask[:, idx] = False
                mask[:, idx+self.veh_dim] = False                  
                veh_depot_attn.append(self.cross_attn(veh, nodes, nodes, key_padding_mask=mask)[0][:,idx,:])
            
            
            
            veh = torch.cat(veh_depot_attn, dim=0).unsqueeze(0) + \
                veh + self.activation(self.ff(veh))

            # veh = self.cross_attn(veh, nodes, nodes, key_padding_mask=node_key_padding_mask)[0] + \
            #     veh + self.activation(self.ff(veh))


        max_veh = torch.zeros((veh.shape[0], 1, veh.shape[-1]), device=veh.device)
        mean_veh = torch.zeros((veh.shape[0], 1, veh.shape[-1]), device=veh.device)
        for idx, (v, m) in enumerate(zip(veh, veh_key_padding_mask)):
            max_veh[idx] = torch.max(v[~m], dim=0)[0]
            mean_veh[idx] = torch.mean(v[~m], dim=0)

        global_veh = torch.cat([max_veh, mean_veh], dim=-1)
                
        return self.global_veh_ff(global_veh), veh, globel_nodes, nodes, node_key_padding_mask

class SelectionEncoder(Module):
    def __init__(self, input_dim=66, embed_dim=64, nhead=4, veh_dim=64, 
                 activation=F.relu, seq_enc='attn', veh_adding_method='cat', 
                 seq_type='embed', embed_slice=True, slice_type='attn', remain_veh_attn=False,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.seq_enc = seq_enc
        self.embed_dim = embed_dim
        if isinstance(activation, str):
            activation = activations[activation]
        
        self.embedding = Embedding_layer(input_dim, embed_dim, 2, 
                                        activation=activation, **self.factory_kwargs)
        if self.seq_enc == 'attn' or self.seq_enc == 'pure_attn':
            self.pe = PositionalEncoding(embed_dim, dropout=0., max_len=500, batch_first=True, **self.factory_kwargs)
            self.seq_encoder = nn.MultiheadAttention(embed_dim, nhead, batch_first=True, **self.factory_kwargs)
        elif self.seq_enc == 'gru' or self.seq_enc == 'pure_gru' or self.seq_enc == 'slice_gru':
            self.seq_encoder = nn.GRU(embed_dim, embed_dim, 1, batch_first=True, **self.factory_kwargs)
            self.seq_veh_ff = Embedding_layer(veh_dim + embed_dim, embed_dim, 2, 
                                               activation=activation, **self.factory_kwargs)
        self.seq_ff = Embedding_layer(embed_dim, embed_dim, 2, activation=activation, **self.factory_kwargs)
        self.seq_norm1 = nn.LayerNorm(embed_dim, **self.factory_kwargs)
        self.seq_norm2 = nn.LayerNorm(embed_dim, **self.factory_kwargs)
        
        self.veh_adding_method = veh_adding_method
        # self.zero_slice = nn.Parameter(torch.zeros((1, self.embed_dim), device=self.factory_kwargs['device']))
        if self.veh_adding_method == 'cat':
            self.slice_embedding = Embedding_layer(veh_dim + embed_dim, embed_dim, 2, 
                                                activation=activation, **self.factory_kwargs)
        elif self.veh_adding_method == 'add':
            assert veh_dim == embed_dim, 'Vehicle dim does not match embed dim'
            self.slice_embedding = Embedding_layer(embed_dim, embed_dim, 2, 
                                                activation=activation, **self.factory_kwargs)

        self.embed_slice = embed_slice
        self.slice_type = slice_type
        if self.embed_slice:
            if self.slice_type == 'attn':
                self.slice_encoder = nn.MultiheadAttention(embed_dim, nhead, batch_first=True, **self.factory_kwargs)
                self.slice_ff = Embedding_layer(embed_dim, embed_dim, 2, activation=activation, **self.factory_kwargs)
                self.slice_norm1 = nn.LayerNorm(embed_dim, **self.factory_kwargs)
            elif self.slice_type == 'ave':
                self.slice_ff = Embedding_layer(2 * embed_dim, embed_dim, 2, activation=activation, **self.factory_kwargs)
            self.slice_norm2 = nn.LayerNorm(embed_dim, **self.factory_kwargs)

        self.seq_type = seq_type

        self.remain_veh_attn = remain_veh_attn
        if self.remain_veh_attn:
            self.remain_veh_attn_block = nn.MultiheadAttention(embed_dim, nhead, batch_first=True, **self.factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def reset_buffer(self, batch_size: int, zero_node: torch.Tensor, veh: torch.Tensor, 
                     veh_all: torch.Tensor, veh_key_padding_mask: torch.Tensor):
        '''
        zero_node: the node No. 0, which usually represents the garbage for vehicles 
                    or the simbol of the end of a nodes' slice, B x 1 x D. 
        veh: the first vehicle.
        veh_all: all veh embeddings, B x M x d.
        veh_key_padding_mask: mask of veh_all B x M.
        '''
        self.zero_node = self.embedding(zero_node)
        if self.seq_enc == 'gru' or self.seq_enc == 'pure_gru' or self.seq_enc == 'slice_gru':
            _, self.zero_node_hidden = self.seq_encoder(self.zero_node)
            self.seq_hidden = torch.zeros_like(self.zero_node_hidden)
        self.zero_embedding = self.seq_block(self.zero_node, 
                                             torch.zeros((batch_size, 1), device=zero_node.device, dtype=torch.bool),
                                             torch.ones(batch_size, device=zero_node.device, dtype=torch.bool))
        if self.seq_enc == 'pure_gru' or self.seq_enc == 'slice_gru':
            return self.zero_embedding, veh + self.remain_veh_attn_block(veh, veh_all, veh_all, veh_key_padding_mask)[0] if self.remain_veh_attn else veh
        if self.seq_enc == 'pure_attn':
            self.seq_buffer = self.zero_node
            return self.zero_embedding, veh + self.remain_veh_attn_block(veh, veh_all, veh_all, veh_key_padding_mask)[0] if self.remain_veh_attn else veh
        self.seq_buffer = [x for x in self.zero_node]
        self.slice_buffer = [torch.zeros((1, self.embed_dim), device=zero_node.device) for _ in range(batch_size)]
        # self.slice_buffer = []
        if self.veh_adding_method == 'cat':
            fusing_feature = torch.cat([self.zero_embedding, veh], dim=-1)
        elif self.veh_adding_method == 'add':
            fusing_feature = self.zero_embedding + veh
        slice_query = self.slice_embedding(fusing_feature)
        if self.embed_slice:
            slice = torch.zeros((batch_size, 1, self.embed_dim), device=zero_node.device)
            slice_embed = self.slice_block(slice_query, 
                                           slice, 
                                           torch.zeros((batch_size, 1), device=zero_node.device).bool())

        if self.seq_type == 'plain':
            return torch.cat([self.zero_embedding, veh], dim=-1), slice_embed if self.embed_slice else None
        return slice_query, slice_embed if self.embed_slice else None

    def seq_block(self, seq: torch.Tensor, seq_mask: Optional[torch.Tensor], key_mask = None):
        '''
        Embedding the current sequence.
        
        seq: current sequence, B x m x D
        seq_mask: B x m

        output:
        seq_embed: current sequence embedding, B x 1 x d
        '''
        if self.seq_enc == 'attn' or self.seq_enc == 'pure_attn':
            seq_embed = self.seq_norm1(seq[:, -1:] + self.seq_encoder(seq[:, -1:], seq, seq, key_padding_mask=seq_mask)[0])
        elif self.seq_enc == 'gru' or self.seq_enc == 'pure_gru' or self.seq_enc == 'slice_gru':
            seq_embed, self.seq_hidden[:, key_mask] = self.seq_encoder(seq[:, -1:], self.seq_hidden[:, key_mask])
            seq_embed = self.seq_norm1(seq[:, -1:] + seq_embed)
        seq_embed = self.seq_norm2(seq_embed + self.seq_ff(seq_embed))
        return seq_embed
    
    def slice_block(self, seq: torch.Tensor, slice: torch.Tensor, slice_mask: torch.Tensor):
        '''
        seq: current sequence embedding, B x 1 x d
        slice: past completed sequece embeddings, B x n x d
        slice_mask: mask of slice, B x n
        '''
        if self.slice_type == 'attn':
            slice_embed = self.slice_norm1(self.slice_encoder(seq, slice, slice, key_padding_mask=slice_mask)[0])
            slice_embed = self.slice_norm2(slice_embed + self.slice_ff(slice_embed))
        elif self.slice_type == 'ave':
            max_slice = torch.zeros((slice.shape[0], 1, slice.shape[-1]), device=slice.device)
            mean_slice = torch.zeros((slice.shape[0], 1, slice.shape[-1]), device=slice.device)
            for idx, (v, m) in enumerate(zip(slice, slice_mask)):
                max_slice[idx] = torch.max(v[~m], dim=0)[0]
                mean_slice[idx] = torch.mean(v[~m], dim=0)

            global_slice = torch.cat([max_slice, mean_slice], dim=-1)
            slice_embed = self.slice_norm2(self.slice_ff(global_slice))
        return slice_embed

    def forward(self, selection: torch.Tensor, veh: torch.Tensor, end_of_seq: torch.Tensor, new_veh: torch.Tensor, 
                key_mask: torch.Tensor, veh_all: torch.Tensor, veh_key_padding_mask: torch.Tensor):
        '''
        selection: new node selected for current sequences, B x 1 x input_dim
        veh: current vehicle, B x 1 x veh_dim
        end_of_seq: bool tensor to note which sequnces have ended, B x 1
        new_veh: new vehicles for ended sequnce, b x 1 x veh_dim, where b is the number of 'True' in end_of_seq
        key_mask: noting which tasks haven't finished yet, B
        veh_all: all veh embeddings, B x M x d.
        veh_key_padding_mask: mask of veh_all B x M.
        '''
        if self.seq_enc == 'attn':
            # 1. Add new selected node to current sequences.
            for idx, (seq, sel, key_end) in enumerate(zip(self.seq_buffer, self.embedding(selection), key_mask)):
                if key_end:
                    self.seq_buffer[idx] = torch.cat([seq, sel], dim=0)
                else:
                    self.seq_buffer[idx] = torch.zeros((1, self.embed_dim), device=seq.device)
                    self.slice_buffer[idx] = torch.zeros((1, self.embed_dim), device=seq.device)
            
            # 2. Transpose the current sequences to masked Tensor for parallel computing.
            seq_mask = [torch.zeros(len(d), device=d.device) for d in self.seq_buffer]
            seq = pad_sequence(self.seq_buffer, batch_first=True)[key_mask]
            seq_mask = pad_sequence(seq_mask, batch_first=True, padding_value=1).bool()[key_mask]

            # 3. Add positional embedding to the sequences
            seq = self.pe(seq)
        
        elif self.seq_enc == 'gru' or self.seq_enc == 'pure_gru' or self.seq_enc == 'slice_gru':
            seq = self.embedding(selection[key_mask])
            if self.seq_enc == 'pure_gru':
                seq = seq + veh[key_mask]
            seq_mask = None
        elif self.seq_enc == 'pure_attn':
            self.seq_buffer = torch.cat([self.seq_buffer, self.embedding(selection) + veh], dim=1)
            seq = self.pe(self.seq_buffer[key_mask])
            seq_mask = torch.zeros((seq.shape[0], seq.shape[1]), device=seq.device, dtype=torch.bool)

        # 4. Get the sequence's feature, where the last nodes appended are used as query.
        seq_embed = torch.zeros((key_mask.shape[0], 1, self.embed_dim), device=seq.device)
        seq_embed[key_mask] = self.seq_block(seq, seq_mask, key_mask)

        # 5. Concat the sequences' feature with their corresponding vehicles' feature, and then embedding them. 
        if self.seq_enc == 'slice_gru':
            seq_embed[end_of_seq] = self.zero_embedding[end_of_seq]
            self.seq_hidden[end_of_seq.unsqueeze(0)] = self.zero_node_hidden[end_of_seq.unsqueeze(0)]
        if self.seq_enc == 'pure_gru' or self.seq_enc == 'slice_gru' or self.seq_enc == 'pure_attn':
            slice_embedding = torch.zeros_like(veh)
            slice_embedding[~end_of_seq] = veh[~end_of_seq]
            slice_embedding[end_of_seq] = new_veh
            return seq_embed[key_mask], slice_embedding[key_mask] + self.remain_veh_attn_block(slice_embedding[key_mask], 
                                                                                               veh_all[key_mask], 
                                                                                               veh_all[key_mask], 
                                                                                               veh_key_padding_mask[key_mask])[0] if self.remain_veh_attn else slice_embedding[key_mask]
        if self.seq_enc == 'attn':
            if self.veh_adding_method == 'cat':
                fusing_feature = torch.cat([seq_embed, veh], dim=-1)
            elif self.veh_adding_method == 'add':
                fusing_feature = seq_embed + veh
        elif self.seq_enc == 'gru':
            if self.veh_adding_method == 'cat':
                fusing_feature = torch.cat([self.seq_hidden.transpose(0, 1), veh], dim=-1)
            elif self.veh_adding_method == 'add':
                fusing_feature = self.seq_hidden.transpose(0, 1) + veh
        seq_with_veh = self.slice_embedding(fusing_feature)

        # 6. If a sequence meets the end, append it into the slice buffer. 
        for idx, embed in zip(torch.arange(len(seq_with_veh), device=end_of_seq.device)[end_of_seq], 
                              seq_with_veh[end_of_seq]):
            # if idx < len(self.slice_buffer):
            self.slice_buffer[idx] = torch.cat([self.slice_buffer[idx], embed], dim=0)
            # else:
            #     self.slice_buffer.insert(idx, embed)
        
        # 7. Use the current sequnces' feature as query, to get slices' encoding.
        #    If the former sequence ends, the query will be the first node in the new sequence(the zero_node)
        slice_query = torch.zeros_like(seq_with_veh)
        slice_query[~end_of_seq] = seq_with_veh[~end_of_seq]
        if torch.any(end_of_seq):
            if self.seq_enc == 'attn':
                if self.veh_adding_method == 'cat':
                    fusing_feature = torch.cat([self.zero_embedding[end_of_seq], new_veh], dim=-1)
                elif self.veh_adding_method == 'add':
                    fusing_feature = self.zero_embedding[end_of_seq] + new_veh
            if self.seq_enc == 'gru':
                if self.veh_adding_method == 'cat':
                    fusing_feature = torch.cat([self.zero_node_hidden.transpose(0, 1)[end_of_seq], new_veh], dim=-1)
                elif self.veh_adding_method == 'add':
                    fusing_feature = self.zero_node_hidden.transpose(0, 1)[end_of_seq] + new_veh
            slice_query[end_of_seq] = self.slice_embedding(fusing_feature)

        if self.embed_slice:
            # 8. Transpose the slices to masked Tensor for parallel computing.
            slice_mask = [torch.zeros(len(d), device=d.device) for d in self.slice_buffer]
            slice = pad_sequence(self.slice_buffer, batch_first=True)[key_mask]
            slice_mask = pad_sequence(slice_mask, batch_first=True, padding_value=1).bool()[key_mask]
            
            # 9. Get the slice ecoding.
            slice_embed = self.slice_block(slice_query[key_mask], slice, slice_mask)

        # 10. Update the sequences ended.
        if self.seq_enc == 'attn':
            for idx, end in enumerate(end_of_seq):
                if end:
                    self.seq_buffer[idx] = self.zero_node[idx]
        elif self.seq_enc == 'gru':
            self.seq_hidden[:, end_of_seq] = self.zero_node_hidden[:, end_of_seq]

        # 11. Return slice queries(current sequences encoding) and slice encoding.
        if self.seq_type == 'plain':
            seq_with_veh = torch.cat([seq_embed, veh], dim=-1)
            if torch.any(end_of_seq):
                seq_with_veh[end_of_seq] = torch.cat([self.zero_embedding[end_of_seq], new_veh], dim=-1)
            return seq_with_veh[key_mask], slice_embed if self.embed_slice else None
        
        if self.seq_enc == 'attn':
            if self.veh_adding_method == 'cat':
                return slice_query[key_mask], slice_embed if self.embed_slice else None
            elif self.veh_adding_method == 'add':
                fusing_feature = seq_embed + veh
                if torch.any(end_of_seq):
                    fusing_feature[end_of_seq] = self.zero_embedding[end_of_seq] + new_veh
                return fusing_feature[key_mask], slice_embed if self.embed_slice else None
        
        seq_with_veh = torch.cat([seq_embed, veh], dim=-1)
        seq_with_veh[end_of_seq] = torch.cat([self.zero_embedding[end_of_seq], new_veh], dim=-1)
        return self.seq_veh_ff(seq_with_veh[key_mask]), slice_embed if self.embed_slice else None


class MaEncoder(Module):
    def __init__(self, target_dim=4, agent_dim=5, enc_type='sa', tgt_enc_type=None, layer_num=1, 
                 agent_layer_num=None, embed_dim=64, nhead=4, activation=F.relu, device=None, 
                 dtype=None, cross_attn_en = False) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(activation, str):
            activation = activations[activation]
        super().__init__()
        self.activation = activation
        self.embed_dim = embed_dim
        
        self.target_embedding = nn.Linear(target_dim, self.embed_dim, bias=False, **self.factory_kwargs)
        self.agent_embedding = nn.Linear(agent_dim, self.embed_dim, bias=False, **self.factory_kwargs)

        if tgt_enc_type is None:
            tgt_enc_type = enc_type       
             
        self.target_attn = FeatureBlock(tgt_enc_type, layer_num, embed_dim, nhead, self.activation, **self.factory_kwargs)
        if agent_layer_num is not None:
            self.agent_attn = FeatureBlock(enc_type, agent_layer_num, embed_dim, nhead, self.activation, **self.factory_kwargs)
        else:
            self.agent_attn = FeatureBlock(enc_type, layer_num, embed_dim, nhead, self.activation, **self.factory_kwargs)            

        self.cross_attn_en = cross_attn_en
        if self.cross_attn_en:
            self.ff = nn.Linear(self.embed_dim, self.embed_dim, **self.factory_kwargs)
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, nhead, batch_first=True, **self.factory_kwargs)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, agent, target, agt_key_padding_mask=None, tgt_key_padding_mask=None):
        agt = self.activation(self.agent_embedding(agent))
        tgt = self.activation(self.target_embedding(target))
        
        tgt = self.target_attn(tgt, tgt_key_padding_mask)
        agt = self.agent_attn(agt, agt_key_padding_mask)

        if self.cross_attn_en:
            agt = self.activation(self.cross_attn(agt, tgt, tgt, key_padding_mask=tgt_key_padding_mask)[0]) + agt + self.activation(self.ff(agt))
                
        return tgt, agt

class PtrEntryActor(Module):
    def __init__(self, query_dim=192, embed_dim=64, nhead=4, activation=F.relu, ptr_fn='attn', pnorm=False, device=None, dtype=None) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(activation, str):
            activation = activations[activation]
        self.activation = activation
        self.embed_dim = embed_dim
        
        self.query_ff = Embedding_layer(query_dim, self.embed_dim, 2, activation=self.activation, **self.factory_kwargs)
        self.query_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0., batch_first=True, **self.factory_kwargs)
        self.query_norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)

        self.ptr_net = MaPtrNet(query_dim=embed_dim, embed_dim=self.embed_dim, ptr_fn=ptr_fn, pnorm=pnorm, **self.factory_kwargs)
        
        self.pnorm = pnorm

        self.entry_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0., batch_first=True, **self.factory_kwargs)
        self.entry_norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)
        self.entry_ff = nn.Sequential(
            nn.Linear(2 * embed_dim, 32, **self.factory_kwargs),
            nn.ReLU(),
            nn.Linear(32, 2, **self.factory_kwargs)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, query, agt, key_padding_mask=None, deterministic: bool = False, tau=1, 
                chosen_idx=None, chosen_entry=None, force_chosen_mask=None):
        """
        Choosing the next node and its corresponding entrance.

        Input:
            agt: encoder output agent embedding, B x N x d
            query: the query, B x 1 x D
            key_padding_mask: specify which agts have been arranged for each batch, B x N
            deterministic: use the most probably choice or sample from the distrbution.
            tau: the temperature factor in gumbel soft max.
            chosen_idx: the chosen index, to get the new prob.
            chosen_entry: the chosen entry, to get the new prob.
            force_chosen_mask: if an element is True, it's corresponding chosen_idx and chosen_entry will be chosen with p(a|s) = 1, B. 
        Output:
            chosen_agt: chosen agts' encoding sampled from agt, concated with the 2d 1-hot vector of entrance, B x 1 x (d + 2).
            choice: the one-hot choice.
            index: chosen index, B x 1.
            chosen_entry: chosen entrance, B x 1.
            prob: probability of the choice, B x 1.
        """
        query = self.query_ff(query)
        query = self.query_norm(query + self.query_attn(query, agt, agt, key_padding_mask)[0])
        
        # 1. Choose the node and corresponding probability p(n|s). 
        node_prob_v = self.ptr_net.dist(query, agt, key_padding_mask, tau)
        # choice, index, prob = self.ptr_net(query, agt, key_padding_mask, deterministic, tau, chosen_idx)        
        
        # Get the chosen nodes' encoding.
        # chosen_agt = agt[choice.detach().bool().squeeze(1)].unsqueeze(1)
        # chosen_agt = self.entry_norm(chosen_agt + self.entry_attn(chosen_agt, agt, agt, key_padding_mask)[0])
        agt_embed = self.entry_norm(agt + self.entry_attn(agt, agt, agt, key_padding_mask)[0])

        entry_feature = torch.cat([query.repeat(1, agt.shape[1], 1), agt_embed], dim=-1)
        if self.pnorm:
            entry_feature = F.normalize(entry_feature, dim=-1, eps=1e-10)
        # # 2. Choose the corresponding entrance.
        entry_prob_v = F.softmax(self.entry_ff(entry_feature) / tau, dim=-1).squeeze(1)
        # entry_prob_v = F.softmax(self.entry_ff(torch.cat([query, chosen_agt], dim=-1)), dim=-1).squeeze(1)
        # if chosen_entry is None:
        #     if deterministic:
        #         chosen_entry = torch.argmax(entry_prob_v, dim=-1, keepdim=True)
        #     else:
        #         entry_dist = torch.distributions.Categorical(entry_prob_v)
        #         chosen_entry = entry_dist.sample([1]).view(-1, 1)
        #     chosen_entry[index==0] = 0

        # # Get the probability of the entrance for the chosen nodes, p(e|n,s).
        # # Force choosing entrance 0 for zero_node.
        # entry_prob = torch.gather(entry_prob_v, dim=-1, index=chosen_entry)
        # entry_prob_1 = torch.ones_like(entry_prob)
        # entry_prob_1[index!=0] = entry_prob[index!=0]

        # # p(a|s) = p(n,e|s) = p(n|s) * p(e|n,s)
        prob_v = node_prob_v.transpose(1, 2).repeat(1, 1, 2) * entry_prob_v
        # p_v = prob_v
        prob_v = prob_v.view((query.shape[0], -1))
        if force_chosen_mask is not None:
            assert chosen_idx is not None and chosen_entry is not None, "chosen_idx and chosen_entry should be given if force_chosen_mask is not None."
            prob_v[force_chosen_mask] = torch.zeros_like(prob_v).scatter_(-1, chosen_idx * 2 + chosen_entry, 1.0)[force_chosen_mask]
        dist = torch.distributions.Categorical(prob_v)
        if chosen_idx is not None and force_chosen_mask is None:
            prob = torch.gather(prob_v, -1, chosen_idx * 2 + chosen_entry)
            index = chosen_idx
        else:
            if deterministic:
                prob, idx = torch.max(prob_v, dim=-1)
                prob = prob.view(-1, 1)
                idx = idx.view(-1, 1)
            else:
                idx = dist.sample([1]).view(-1, 1)
                prob = torch.gather(prob_v, dim=-1, index=idx)
            index = idx // 2
            chosen_entry = idx % 2

        choice = torch.zeros_like(node_prob_v, memory_format=torch.legacy_contiguous_format).scatter_(-1, index.unsqueeze(-1), 1.0)

        # 3. Concate the chosen nodes' encoding and the corredponding entrance one-hot vector.
        entry_choice = torch.zeros((query.shape[0], 2), device=query.device).scatter_(-1, chosen_entry, 1.0).unsqueeze(1)
        chosen_agt = agt[choice.detach().bool().squeeze(1)].unsqueeze(1)
        chosen_agt = torch.cat([chosen_agt, entry_choice], dim=-1)

        return chosen_agt, choice, index, chosen_entry, prob, prob_v

class StepCritic(Module):
    def __init__(self, query_dim=192, embed_dim=64, nhead=4, activation=F.relu, device=None, dtype=None) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(activation, str):
            activation = activations[activation]
        self.activation = activation
        self.embed_dim = embed_dim
        
        self.query_ff = Embedding_layer(query_dim, self.embed_dim, 2, activation=self.activation, **self.factory_kwargs)
        self.query_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0., batch_first=True, **self.factory_kwargs)
        self.query_norm = nn.LayerNorm(self.embed_dim, **self.factory_kwargs)

        self.critic_ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, **self.factory_kwargs),
            nn.ReLU(),
            nn.Linear(embed_dim, 32, **self.factory_kwargs),
            nn.ReLU(),
            nn.Linear(32, 1, **self.factory_kwargs)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, query, node, key_padding_mask=None) -> torch.Tensor:
        """
        node: encoder output node embedding, B x N x d
        query: the query, B x 1 x D
        key_padding_mask: specify which agts have been arranged for each batch, B x N
        """
        query = self.query_ff(query)
        query = self.query_norm(query + self.query_attn(query, node, node, key_padding_mask)[0])

        return self.critic_ff(query)

class GNNAC(Module):
    default_common = {
        'embed_dim': 64,
        'activation': 'relu',
    }

    default_enc = {
        'gnn_cfg': {
            'gnn_cfg': {
                'node_dim': 3, 
                'edge_dim': 4, 
                'embedding_layer': 1,
                'nhead': 4, 
                'layer_num': 2, 
                'dropout': 0.1,
                'whole_batch_edge_processing': False, 
                'edge_norm': True,
            },
            'gff_cfg': {
                'layer': 2
            }
        },
        'veh_dim': 5, 
        'embedding_layer': 1, 
        'enc_type': 'enc', 
        'layer_num': 1, 
        'nhead': 4, 
        'cross_attn_en': False
    }

    default_selection_enc = {
        'input_dim': 66, 
        'nhead': 4, 
        'veh_dim': 64,
        'seq_enc': 'pure_gru',
        'veh_adding_method': 'cat',
        'seq_type': 'embed',
        'embed_slice': True,
        'slice_type': 'attn',
        'remain_veh_attn': False
    }

    default_actor = {
        'query_dim': 256, 
        'nhead': 4,
        'ptr_fn': 'attn',
        'pnorm': False
    }

    default_critic = {
        'query_dim': 256, 
        'nhead': 4,
    }

    default_critic_1step = {
        'layer_num': 2
    }

    def __init__(self, common_cfg=None, encoder_cfg=None, sel_encoder_cfg=None, actor_cfg=None, critic_cfg=None, sequential_sel=True,
                 single_step=False, critic_list=['default'], critic_detach=False, device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.single_step = single_step
        self.critic_detach = critic_detach
        self.common = deepcopy(self.default_common)
        self.encoder_cfg = deepcopy(self.default_enc)
        self.selection_enc = deepcopy(self.default_selection_enc)
        self.actor_cfg = deepcopy(self.default_actor)
        self.critic_cfg = deepcopy(self.default_critic_1step if self.single_step else self.default_critic)
        self.sequential_sel = sequential_sel

        if common_cfg is not None:
            self.common.update(common_cfg)
        if encoder_cfg is not None:
            self.encoder_cfg.update(encoder_cfg)
        if sel_encoder_cfg is not None:
            self.selection_enc.update(sel_encoder_cfg)
        if actor_cfg is not None:
            self.actor_cfg.update(actor_cfg)
        if critic_cfg is not None:
            self.critic_cfg.update(critic_cfg)

        self.encoder = MaGNNEncoder(self.common, **self.encoder_cfg, **self.factory_kwargs)        
        self.sel_enc = SelectionEncoder(**self.common, **self.selection_enc, **self.factory_kwargs)         
        self.actor = PtrEntryActor(**self.common, **self.actor_cfg, **self.factory_kwargs)
        self.critic = nn.ModuleDict({})
        for name in critic_list:
            if self.single_step:
                self.critic.update({name: MLPDecoder(**self.common, **self.critic_cfg, **self.factory_kwargs)})
            else:
                self.critic.update({name: StepCritic(**self.common, **self.critic_cfg, **self.factory_kwargs)})
        
        self.tau = 1

        self.cfg = {
            'single_step': self.single_step,
            'critic_detach': self.critic_detach,
            'common_cfg': self.common,
            'encoder_cfg': self.encoder_cfg,
            'sel_encoder_cfg': self.selection_enc,
            'actor_cfg': self.actor_cfg,
            'critic_cfg': self.critic_cfg,
            'critic_list': critic_list
        }

        self.actor_param = nn.ModuleList([self.encoder, self.sel_enc, self.actor])
        self.actor_param_without_gnn = nn.ModuleList([self.encoder.own_modules, self.sel_enc, self.actor])
        if self.critic_detach:
            self.critic_param = nn.ModuleList([self.critic])
        else:
            self.critic_param = nn.ModuleList([self.encoder, self.sel_enc, self.critic])

    def forward(self, data, info, deterministic: bool = False, chosen_idx=None, chosen_entry=None, 
                actor_grad: bool = True, criticize: bool = True, dist_only: bool = False, 
                criticize_only: bool = False, force_chosen: bool=False):
        '''
        data: {'graph': PyG Graph Data, 'vehicles': B x M x 5 Tensor}
        info: {'veh_key_padding_mask': B x M Tensor, 'num_veh': B x 1 tensor}
        deterministic: if True, output the choice with max probability.
        chosen_idx: chosen indexs.
        chosen_entry: chosen entry.
        actor_grad: whether need the actor's gradiant.
        criticize: whether calculate V(s).
        dist_only: if True, only out put the distribution.
        criticize only: if True, only calculate V(s).
        force_chosen: if True, the choices in chosen_idx(index != -1) will be chosen with probability 1.0. 
        '''
        assert not (dist_only and criticize_only)
        if dist_only:
            criticize = False
        if criticize_only:
            criticize = True
            actor_grad = False

        veh_key_padding_mask = torch.zeros(data['vehicles'].shape[:2], dtype=torch.bool, device=data['vehicles'].device) \
                                if info['veh_key_padding_mask'] is None else info['veh_key_padding_mask'].clone()
        
        # Encode
        global_veh, veh, globel_nodes, nodes, node_key_padding_mask = self.encoder(data['graph'], 
                                                                                   data['vehicles'], 
                                                                                   veh_key_padding_mask)
        
        N = nodes.shape[1]
        M = veh.shape[1]

        # Decode Sequence
        bsz = veh.shape[0]

        value = {}
        if self.single_step and criticize_only:
            value_mask = torch.zeros((bsz, 2), dtype=torch.bool, device=veh.device)
            value_mask[:, 0] = True
            global_feature = torch.cat([globel_nodes, global_veh], dim=-1).squeeze(1)
            if self.critic_detach:
                global_feature = global_feature.detach()
            for key, mod in self.critic.items():
                value.update({key: torch.zeros((bsz, 2), device=veh.device)})
                value[key][:, 0:1] = mod(global_feature)
            return value, value_mask

        veh_idx = torch.zeros((bsz, 1), dtype=int, device=info['num_veh'].device)
        end_of_seq = torch.zeros((bsz), dtype=torch.bool, device=info['num_veh'].device)
        key_mask = torch.ones((bsz), dtype=torch.bool)
        node_key_padding_mask[:, 0:1] = (veh_idx >= info['num_veh'] - 1)
        
        zero_node = torch.cat([nodes[:, 0:1], 
                               torch.ones((bsz, 1, 1), device=nodes.device), 
                               torch.zeros((bsz, 1, 1), device=nodes.device)], 
                               dim=-1)

        node_key_padding_mask[:, 0:1] = True
        
        start_mask = node_key_padding_mask.clone()[:, :M]
        end_mask = node_key_padding_mask.clone()[:, M:2*M]
        node_mask = node_key_padding_mask.clone()[:, 2*M:]
        
        start_mask[:, :] = True
        end_mask[:, :] = True
        end_mask[:, 0:1] = False

        input_mask = torch.cat((start_mask, end_mask, node_mask), dim=1)
        
        seq_embed, slice_embed = self.sel_enc.reset_buffer(bsz, zero_node, veh[:, 0:1], veh, veh_key_padding_mask)

        step = 0

        # N-1 nodes
        choice = torch.zeros((bsz, N - 1, N), device=veh.device)
        index = - torch.ones((bsz, N - 1), device=veh.device, dtype=torch.long)
        entry = torch.zeros((bsz, N - 1), device=veh.device, dtype=torch.long)
        prob = torch.ones((bsz, N - 1), device=veh.device)
        dists = torch.ones((bsz, N - 1, 2 * N), device=veh.device)

        value_mask = torch.zeros((bsz, N), dtype=torch.bool, device=veh.device)

        for name in self.critic.keys():
            if self.single_step:
                value.update({name: torch.zeros((bsz, 2), device=veh.device)})
            else:
                value.update({name: torch.zeros((bsz, N), device=veh.device)})
            
        seq_enc = [{'tar':[], 'split': []} for _ in range(bsz)]

        if force_chosen: 
            for bsz_idx in range(bsz):
                for arrange in chosen_idx[bsz_idx]:
                    for node_idx in arrange:
                        node_key_padding_mask[bsz_idx][node_idx] = True
            idx_in_seq = torch.zeros((bsz, 1), dtype=int, device=info['num_veh'].device)

        while torch.any(key_mask):
            if self.sel_enc.embed_slice:
                query = torch.cat([global_veh[key_mask], globel_nodes[key_mask], seq_embed, slice_embed], dim=-1)
            else:
                query = torch.cat([global_veh[key_mask], globel_nodes[key_mask], seq_embed], dim=-1)

            if not force_chosen: 
                force_chosen_mask = None
                cur_chosen_idx = chosen_idx[key_mask, step:step+1] if chosen_idx is not None else None
                cur_chosen_entry = chosen_entry[key_mask, step:step+1] if chosen_entry is not None else None
            else:
                force_chosen_mask = torch.zeros_like(key_mask).bool()
                cur_chosen_idx = torch.zeros((bsz, 1), dtype=torch.long, device=veh.device)
                cur_chosen_entry = torch.zeros((bsz, 1), dtype=torch.long, device=veh.device)
                for bsz_idx, (car_idx, car_idx_in_seq, mask) in enumerate(zip(veh_idx, idx_in_seq, key_mask)):
                    if not mask:
                        continue
                    if len(chosen_idx[bsz_idx][car_idx]) > car_idx_in_seq:
                        cur_chosen_idx[bsz_idx] = chosen_idx[bsz_idx][car_idx][car_idx_in_seq]
                        cur_chosen_entry[bsz_idx] = chosen_entry[bsz_idx][car_idx][car_idx_in_seq]
                        force_chosen_mask[bsz_idx] = True
                cur_chosen_idx = cur_chosen_idx[key_mask]
                cur_chosen_entry = cur_chosen_entry[key_mask]
                force_chosen_mask = force_chosen_mask[key_mask]

            if actor_grad:
                chosen_agt, cur_choice, cur_index, cur_entry, cur_prob, cur_dist = self.actor(query, nodes[key_mask], 
                                                                        input_mask[key_mask], 
                                                                        deterministic, self.tau, 
                                                                        cur_chosen_idx, 
                                                                        cur_chosen_entry,
                                                                        force_chosen_mask)
            else:
                with torch.no_grad():
                    chosen_agt, cur_choice, cur_index, cur_entry, cur_prob, cur_dist = self.actor(query, nodes[key_mask], 
                                                                        input_mask[key_mask], 
                                                                        deterministic, self.tau, 
                                                                        cur_chosen_idx, 
                                                                        cur_chosen_entry,
                                                                        force_chosen_mask)

            choice[key_mask, step:step+1] = cur_choice
            index[key_mask, step:step+1] = cur_index
            prob[key_mask, step:step+1] = cur_prob
            entry[key_mask, step:step+1] = cur_entry
            dists[key_mask, step] = cur_dist
            value_mask[:, step] = key_mask
            
            if criticize and not self.single_step:
                for key, mod in self.critic.items():
                    if self.critic_detach:
                        value[key][key_mask, step:step+1] = mod(query.detach(), nodes[key_mask].detach(), input_mask[key_mask]).squeeze(1)
                    else:
                        value[key][key_mask, step:step+1] = mod(query, nodes[key_mask], input_mask[key_mask]).squeeze(1)

            node_key_padding_mask[choice[:, step].detach().bool()] = True
            start_mask = node_key_padding_mask.clone()[:, :M]
            end_mask = node_key_padding_mask.clone()[:, M:2*M]
            node_mask = node_key_padding_mask.clone()[:, 2*M:]
            start_of_seq = torch.zeros((bsz), dtype=torch.bool, device=info['num_veh'].device)
            for bidx, (gen, idx, ent) in enumerate(zip(seq_enc, index[:, step], entry[:, step])):
                if key_mask[bidx]:
                    if idx.item() == veh_idx[bidx].item() + M:
                        end_of_seq[bidx] = True
                        end_mask[bidx, :] = True
                        node_mask[bidx, :] = True
                    elif end_of_seq[bidx]:
                        end_of_seq[bidx] = False
                        start_mask[bidx, :] = True
                        end_mask[bidx, :] = True
                        end_mask[bidx, idx.item()] = False
                        gen['split'].append([len(gen['tar']), veh_idx[bidx].item()])
                        veh_idx[bidx] = idx.item()
                        start_of_seq[bidx] = True
                    elif idx.item() >= 2*M:
                        if torch.all(start_mask[bidx]):
                            end_mask[bidx, :] = True
                            if torch.all(node_mask[bidx]):
                                end_mask[bidx, veh_idx[bidx].item()] = False
                        else:
                            start_mask[bidx, :] = True
                            end_mask[bidx, :] = True
                            end_mask[bidx, veh_idx[bidx].item()] = False
                        gen['tar'].append([idx.item() - 2 * M, ent.item()])

                    if torch.all(node_key_padding_mask[bidx]):
                        gen['split'].append([len(gen['tar']), veh_idx[bidx].item()])

            input_mask = torch.cat((start_mask, end_mask, node_mask), dim=1)

            cur_veh = torch.zeros((bsz, 1, veh.shape[-1]), dtype=veh.dtype, device=veh.device)
            cur_veh[key_mask] = torch.gather(veh[key_mask], index=veh_idx[key_mask].repeat(1, veh.shape[-1]).unsqueeze(1), dim=1)
            selection = torch.zeros((bsz, 1, chosen_agt.shape[-1]), device=chosen_agt.device)
            selection[key_mask] = chosen_agt
            
            if force_chosen:
                idx_in_seq = idx_in_seq + 1
                idx_in_seq[start_of_seq] = 0
            new_veh = torch.gather(veh[start_of_seq], index=veh_idx[start_of_seq].repeat(1, veh.shape[-1]).unsqueeze(1), dim=1)      
            key_mask = ~torch.all(node_key_padding_mask, dim=-1)
            
            # TODO:change here
            # if self.sel_enc.remain_veh_attn:
            #     for mask_idx, count in enumerate(veh_count):
            #         if end_of_seq[mask_idx]:
            #             veh_key_padding_mask[mask_idx, count-1]= True
                        
            seq_embed, slice_embed = self.sel_enc(selection, cur_veh, start_of_seq, new_veh, key_mask, veh, veh_key_padding_mask)

            step += 1

            if step>=N-2:
                print(111)

        dist = torch.distributions.Categorical(dists[value_mask[:, :-1]])
        if self.single_step:
            value_mask = torch.zeros((bsz, 2), dtype=torch.bool, device=veh.device)
            value_mask[:, 0] = True
            prob = torch.exp(torch.sum(torch.log(prob), dim=-1, keepdim=True)) + 1e-12
            if criticize:
                global_feature = torch.cat([globel_nodes, global_veh], dim=-1).squeeze(1)
                if self.critic_detach:
                    global_feature = global_feature.detach()
                for key, mod in self.critic.items():
                    value[key][:, 0:1] = mod(global_feature)
        
        if criticize_only:
            return value, value_mask
        if dist_only:
            return dist
        if criticize:
            return seq_enc, choice, index, entry, prob, dist, value, value_mask
        return seq_enc, choice, index, entry, prob, dist, value_mask


class MaDecoder(Module):
    def __init__(self, query_dim=192, embed_dim=64, nhead=4, activation=F.relu, ptr_fn='attn', device=None, dtype=None, **kwargs) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(activation, str):
            activation = activations[activation]
        super().__init__()
        self.activation = activation
        self.embed_dim = embed_dim
        self.activation = activation
        self.decoder_ff = nn.Linear(query_dim, self.embed_dim, **self.factory_kwargs)
        # self.self_attn = FeatureBlock(enc_type, layer_num, self.embed_dim, nhead, self.activation, **self.factory_kwargs)
        self.self_attn = nn.MultiheadAttention(self.embed_dim, nhead, dropout=0.1, batch_first=True, **self.factory_kwargs)
        self.ptr_net = MaPtrNet(query_dim=embed_dim, embed_dim=self.embed_dim, ptr_fn=ptr_fn, **self.factory_kwargs)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, query, agt, key_mask=None, key_padding_mask=None, zero_attn_mask=None, deterministic: bool = False, tau=1, idx=None):
        """
        agt: encoder output agent embedding, B x N x d
        query: the query, B x 1 x D
        tgt_select: specify which target is chosen for each batch, B x M
        key_mask: specify which samples are not arranged yet, B
        key_padding_mask: specify which agts have been arranged for each batch, B x N
        zero_attn_mask: B x 1
        deterministic: use the most probably choice or sample from the distrbution.
        tau: the temperature factor in gumbel soft max.
        idx: the chosen index, to get the new prob.
        """
        bsz = agt.shape[0]
        N = agt.shape[1]  
        if key_mask is None:
            key_mask = torch.ones(bsz, dtype=torch.bool, device=key_padding_mask.device)

        query = self.activation(self.decoder_ff(query))[key_mask]
        kv = torch.cat([torch.zeros(agt.shape[0], 1, agt.shape[2], dtype=agt.dtype, device=agt.device), agt], dim=1)

        if zero_attn_mask is None:
            zero_attn_mask = torch.zeros((bsz, 1), dtype=torch.bool, device=key_padding_mask.device)
        ptr_key_padding_mask = torch.cat([zero_attn_mask, key_padding_mask], dim=1)[key_mask]
        query = query + self.self_attn(query, kv[key_mask], kv[key_mask], ptr_key_padding_mask)[0]
        
        choice, index, prob = self.ptr_net(query, agt[key_mask], ptr_key_padding_mask, deterministic, tau, idx)        
        
        real_choice = torch.zeros((bsz, 1, N + 1), dtype=choice.dtype, device=choice.device)
        real_index = torch.zeros((bsz, 1), dtype=index.dtype, device=index.device)
        real_prob = torch.ones((bsz, 1), dtype=prob.dtype, device=prob.device)

        real_choice[key_mask] = choice
        real_index[key_mask] = index
        real_prob[key_mask] = prob

        return real_choice, real_index, real_prob

class MaPtrActor(Module):
    default_common = {
        'embed_dim': 64,
        'activation': F.relu
    }

    default_enc = {
        'target_dim': 4,
        'agent_dim': 5,
        'enc_type': 'enc',
        'layer_num': 1,
        'nhead': 4
    }

    default_dec = {
        'layer_num': 1, 
        'enc_type': 'sa',
        'query_dim': 256, 
        'nhead': 4,
        'ptr_fn': 'attn'
    }

    def __init__(self, common_cfg=None, encoder_cfg=None, decoder_cfg=None, device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.common = deepcopy(self.default_common)
        self.enc = deepcopy(self.default_enc)
        self.dec = deepcopy(self.default_dec)

        if common_cfg is not None:
            self.common.update(common_cfg)
        if encoder_cfg is not None:
            self.enc.update(encoder_cfg)
        if decoder_cfg is not None:
            self.dec.update(decoder_cfg)
                
        self.encoder = MaEncoder(**self.common, **self.enc, **self.factory_kwargs)         
        self.decoder = MaDecoder(**self.common, **self.dec, **self.factory_kwargs)
        

    def forward(self, data, info, feature_only= False, deterministic: bool = False, tau=1, chosen_idx = None):
        '''
        data: {'agents': B x N x 5 Tensor, 'targets': B x M x 4 Tensor}
        info: {'agt_key_padding_mask': B x N Tensor, 'tgt_key_padding_mask': B x M Tensor, 'num_tgt': B x 1 tensor}
        '''
        agt_key_padding_mask = torch.zeros(data['agents'].shape[:2], dtype=torch.bool, device=self.factory_kwargs['device']) if info['agt_key_padding_mask'] is None else info['agt_key_padding_mask'].clone()
        tgt_key_padding_mask = torch.zeros(data['targets'].shape[:2], dtype=torch.bool, device=self.factory_kwargs['device']) if info['tgt_key_padding_mask'] is None else info['tgt_key_padding_mask'].clone()
        
        # Encode
        tgt, agt = self.encoder(data['agents'], data['targets'], agt_key_padding_mask, tgt_key_padding_mask)

        if feature_only:
            return tgt, agt
        
        # Decode Sequence
        bsz = data['agents'].shape[0]

        avg_tgt = torch.zeros((tgt.shape[0], tgt.shape[-1]), dtype=tgt.dtype, device=tgt.device)
        avg_agt = torch.zeros((agt.shape[0], agt.shape[-1]), dtype=agt.dtype, device=agt.device)

        for idx, t, mask in zip(range(tgt.shape[0]), tgt, tgt_key_padding_mask):
            avg_tgt[idx] = torch.mean(t[~mask], dim=0)
        for idx, a, mask in zip(range(agt.shape[0]), agt, agt_key_padding_mask):
            avg_agt[idx] = torch.mean(a[~mask], dim=0)

        tgt_count = torch.zeros((bsz, 1), dtype=int, device=info['num_tgt'].device)
        key_mask = (tgt_count < info['num_tgt']).squeeze(-1)
        zero_attn_mask = (tgt_count >= info['num_tgt'])
        
        agt_count = torch.zeros((bsz, 1), device=agt.device)
        cur_agt_ave = torch.zeros_like(avg_agt, device=avg_agt.device)

        output_choice = None
        output_index = None
        output_prob = None

        seq_enc = [{'tar':[], 'split': []} for _ in range(bsz)]

        while torch.any(key_mask):
            selected_target = torch.zeros((bsz, tgt.shape[-1]), dtype=tgt.dtype, device=tgt.device)
            selected_target[key_mask] = torch.gather(tgt[key_mask], index=tgt_count[key_mask].repeat(1, tgt.shape[-1]).unsqueeze(1), dim=1).squeeze(1)
            query = torch.cat([avg_tgt, avg_agt, selected_target, cur_agt_ave], dim=1).unsqueeze(1)
            
            choice, index, prob = self.decoder(query, agt, key_mask, agt_key_padding_mask, zero_attn_mask, deterministic, tau)

            output_choice = torch.cat([output_choice, choice], dim=1) if output_choice is not None else choice
            output_index = torch.cat([output_index, index], dim=1) if output_index is not None else index
            output_prob = torch.cat([output_prob, prob], dim=1) if output_prob is not None else prob
            
            for bidx, (gen, idx) in enumerate(zip(seq_enc, index)):
                if key_mask[bidx]:
                    if idx.item() == 0:
                        gen['split'].append(len(gen['tar']))
                    else:
                        gen['tar'].append(idx.item() - 1)
            
            agt_key_padding_mask[choice.detach().bool()[:, -1, 1:]] = True

            end_of_target = (index == 0).flatten()
            tgt_count = tgt_count + end_of_target.unsqueeze(1).int()
            key_mask = (tgt_count < info['num_tgt']).squeeze(-1)
            zero_attn_mask = (tgt_count >= info['num_tgt'])

            cur_agt_ave[~end_of_target] = cur_agt_ave[~end_of_target].clone() + agt[choice.detach().bool()[:, -1, 1:]]
            # cur_agt_ave[~end_of_target] = cur_agt_ave[~end_of_target].clone() * agt_count[~end_of_target] + agt[choice.detach().bool()[:, -1, 1:]]
            # agt_count = agt_count + 1
            # cur_agt_ave = cur_agt_ave / agt_count
            cur_agt_ave = cur_agt_ave * (1 - end_of_target.float().view(-1, 1))
            # agt_count = agt_count * (1 - end_of_target.float().view(-1, 1))

        return seq_enc, output_choice, output_index, output_prob

class PtrActor(Module):
    default_common = {
        'embed_dim': 64,
        'activation': F.relu
    }

    default_enc = {
        'target_dim': 4,
        'agent_dim': 9,
        'enc_type': 'enc',
        'layer_num': 1,
        'nhead': 4,
        'cross_attn_en': False
    }

    default_dec = {
        'layer_num': 1, 
        'enc_type': 'sa',
        'query_dim': 192, 
        'nhead': 4
    }

    def __init__(self, common_cfg=None, encoder_cfg=None, decoder_cfg=None, device=None, dtype=None) -> None:
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.common = deepcopy(self.default_common)
        self.enc = deepcopy(self.default_enc)
        self.dec = deepcopy(self.default_dec)

        if common_cfg is not None:
            self.common.update(common_cfg)
        if encoder_cfg is not None:
            self.enc.update(encoder_cfg)
        if decoder_cfg is not None:
            self.dec.update(decoder_cfg)
                
        self.encoder = MaEncoder(**self.common, **self.enc, **self.factory_kwargs)         
        self.decoder = MaDecoder(**self.common, **self.dec, **self.factory_kwargs)
        

    def forward(self, data, info, feature_only= False, deterministic: bool = False, tau=1, choice=None):
        '''
        data: {'agents': B x N x 5 Tensor, 'targets': B x M x 4 Tensor}
        info: {'agt_key_padding_mask': B x N Tensor, 'tgt_key_padding_mask': B x M Tensor, 'num_tgt': B x 1 tensor}
        '''
        agt_key_padding_mask = torch.zeros(data['agents'].shape[:2], dtype=torch.bool, device=self.factory_kwargs['device']) if info['agt_key_padding_mask'] is None else info['agt_key_padding_mask'].clone()
        tgt_key_padding_mask = torch.zeros(data['targets'].shape[:2], dtype=torch.bool, device=self.factory_kwargs['device']) if info['tgt_key_padding_mask'] is None else info['tgt_key_padding_mask'].clone()
        
        # Encode
        tgt, agt = self.encoder(data['agents'], data['targets'], agt_key_padding_mask, tgt_key_padding_mask)

        if feature_only:
            return tgt, agt
        
        # Decode Sequence
        avg_tgt = torch.zeros((tgt.shape[0], tgt.shape[-1]), dtype=tgt.dtype, device=tgt.device)
        avg_agt = torch.zeros((agt.shape[0], agt.shape[-1]), dtype=agt.dtype, device=agt.device)

        for idx, t, mask in zip(range(tgt.shape[0]), tgt, tgt_key_padding_mask):
            avg_tgt[idx] = torch.mean(t[~mask], dim=0)
        for idx, a, mask in zip(range(agt.shape[0]), agt, agt_key_padding_mask):
            avg_agt[idx] = torch.mean(a[~mask], dim=0)
       
        self_agt = agt[:, 0]
        
        query = torch.cat([avg_tgt, avg_agt, self_agt], dim=1).unsqueeze(1)
        
        bsz = tgt.shape[0]
        zero_attn_mask = torch.zeros((bsz, 1), dtype=torch.bool, device=tgt_key_padding_mask.device)

        return self.decoder(query, tgt, key_padding_mask=tgt_key_padding_mask, zero_attn_mask=zero_attn_mask, deterministic=deterministic, tau=tau, idx=choice)

class MLPDecoder(Module):
    def __init__(self, embed_dim=64, layer_num=2, activation=F.relu, device=None, dtype=None) -> None:
        super().__init__()
        if isinstance(activation, str):
            activation = activations[activation]
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.activation = activation
        if layer_num == 0:
            self.mlp = []
            self.last_layer = [nn.Linear(2 * embed_dim, 1, **self.factory_kwargs)]
        else:
            first_layer = nn.Linear(2 * embed_dim, embed_dim, **self.factory_kwargs)
            self.last_layer = nn.Linear(embed_dim, 1, **self.factory_kwargs)
            self.mlp = [first_layer] + [nn.Linear(embed_dim, embed_dim, **self.factory_kwargs) for _ in range(layer_num-1)]
            self.mlp = nn.ModuleList(self.mlp)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        for mod in self.mlp:
            x = self.activation(mod(x))
        return self.last_layer(x)

class MaCritic(Module):
    default_common = {
        'embed_dim': 64,
        'activation': F.relu
    }

    default_enc = {
        'target_dim': 4,
        'agent_dim': 5,
        'layer_num': 1,
        'nhead': 4
    }

    default_dec = {
        'layer_num': 1, 
    }

    def __init__(self, common_cfg=None, encoder_cfg=None, decoder_cfg=None, share_enc=False, device=None, dtype=None) -> None:
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.common = deepcopy(self.default_common)
        self.enc = deepcopy(self.default_enc)
        self.dec = deepcopy(self.default_dec)

        if common_cfg is not None:
            self.common.update(common_cfg)
        if encoder_cfg is not None:
            self.enc.update(encoder_cfg)
        if decoder_cfg is not None:
            self.dec.update(decoder_cfg)
        
        self.share_enc = share_enc
        if not self.share_enc:
            self.encoder = MaEncoder(**self.common, **self.enc, **self.factory_kwargs)         
        self.decoder = MLPDecoder(**self.common, **self.dec, **self.factory_kwargs)
        

    def forward(self, data, info):
        '''
        data: {'agent': B x N x 5 Tensor, 'target': B x M x 4 Tensor}
        info: {'agt_key_padding_mask': B x N Tensor, 'tgt_key_padding_mask': B x M Tensor, 'num_tgt': B x 1 tensor}
        '''
        if not self.share_enc:
            # Encode
            agt_key_padding_mask = torch.zeros(data['agents'].shape[:2], dtype=torch.bool, device=self.factory_kwargs['device']) if info['agt_key_padding_mask'] is None else info['agt_key_padding_mask'].clone()
            tgt_key_padding_mask = torch.zeros(data['targets'].shape[:2], dtype=torch.bool, device=self.factory_kwargs['device']) if info['tgt_key_padding_mask'] is None else info['tgt_key_padding_mask'].clone()
            tgt, agt = self.encoder(data['agents'], data['targets'], agt_key_padding_mask, tgt_key_padding_mask)
        else:
            agt_key_padding_mask = torch.zeros(data[1].shape[:2], dtype=torch.bool, device=self.factory_kwargs['device']) if info['agt_key_padding_mask'] is None else info['agt_key_padding_mask'].clone()
            tgt_key_padding_mask = torch.zeros(data[0].shape[:2], dtype=torch.bool, device=self.factory_kwargs['device']) if info['tgt_key_padding_mask'] is None else info['tgt_key_padding_mask'].clone()
            tgt, agt = data

        avg_tgt = torch.zeros((tgt.shape[0], tgt.shape[-1]), dtype=tgt.dtype, device=tgt.device)
        avg_agt = torch.zeros((agt.shape[0], agt.shape[-1]), dtype=agt.dtype, device=agt.device)

        for idx, t, mask in zip(range(tgt.shape[0]), tgt, tgt_key_padding_mask):
            avg_tgt[idx] = torch.sum(t[~mask], dim=0)
        for idx, a, mask in zip(range(agt.shape[0]), agt, agt_key_padding_mask):
            avg_agt[idx] = torch.sum(a[~mask], dim=0)

        return self.decoder(torch.cat([avg_tgt, avg_agt], dim=-1))
    
class MaPtrActorCritic(Module):
    default = {
        'actor_cfg': {
            'common_cfg': {
                'embed_dim': 64,
                'activation': F.relu
            },

            'encoder_cfg': {
                'target_dim': 4,
                'agent_dim': 5,
                'enc_type': 'enc',
                'layer_num': 1,
                'nhead': 4,
                'cross_attn_en': False
            },

            'decoder_cfg': {
                'layer_num': 1, 
                'enc_type': 'sa',
                'query_dim': 256, 
                'nhead': 4,
                'ptr_fn': 'ff'
            }
        },
        'critic_cfg': {
            'common_cfg': {
                'embed_dim': 64,
                'activation': F.relu
            },

            'encoder_cfg': {
                'target_dim': 4,
                'agent_dim': 5,
                'enc_type': 'enc',
                'layer_num': 1,
                'nhead': 4,
                'cross_attn_en': False
            },

            'decoder_cfg': {
                'layer_num': 1, 
            }
        },
        'share_feature': True,
        'dtype': torch.float32
    }

    def __init__(self, cfg=None, device=None) -> None:
        super().__init__()
        self.cfg = deepcopy(self.default)
        if cfg is not None:
            self.cfg.update(cfg)
        self.training = True
        self.actor = MaPtrActor(**self.cfg['actor_cfg'], dtype=self.cfg['dtype'], device=device)
        self.critic = MaCritic(**self.cfg['critic_cfg'], share_enc=self.cfg['share_feature'], dtype=self.cfg['dtype'], device=device)
        self.cfg.pop('dtype')

    def forward(self, data, info):
        self.eval()
        with torch.no_grad():
            seq_enc, choice, index, prob = self.act(data, info)
            V = self.criticize(data, info)
        self.train()
        return seq_enc, choice, index, prob, V
    
    def act(self, data, info):
        return self.actor(data, info, deterministic=True)

    def sample(self, data, info, tau=1):
        return self.actor(data, info, tau=tau)

    def criticize(self, data, info):
        if self.cfg['share_feature']:
            tgt, agt = self.actor(data, info, feature_only=True)
            return self.critic((tgt, agt), info)
        else:
            return self.critic(data, info)

class PtrActorCritic(Module):
    default = {
        'actor_cfg': {
            'common_cfg': {
                'embed_dim': 64,
                'activation': F.relu
            },

            'encoder_cfg': {
                'target_dim': 4,
                'agent_dim': 9,
                'enc_type': 'enc',
                'layer_num': 1,
                'nhead': 4,
                'cross_attn_en': False
            },

            'decoder_cfg': {
                'layer_num': 1, 
                'enc_type': 'sa',
                'query_dim': 192, 
                'nhead': 4
            }
        },
        'critic_cfg': {
            'common_cfg': {
                'embed_dim': 64,
                'activation': F.relu
            },

            'encoder_cfg': {
                'target_dim': 4,
                'agent_dim': 9,
                'enc_type': 'enc',
                'layer_num': 1,
                'nhead': 4,
                'cross_attn_en': False
            },

            'decoder_cfg': {
                'layer_num': 1, 
            }
        },
        'share_feature': False,
        'dtype': torch.float32
    }

    def __init__(self, cfg=None, device=None) -> None:
        super().__init__()
        self.cfg = deepcopy(self.default)
        if cfg is not None:
            self.cfg.update(cfg)
        self.training = True
        self.actor = PtrActor(**self.cfg['actor_cfg'], dtype=self.cfg['dtype'], device=device)
        self.critic = MaCritic(**self.cfg['critic_cfg'], share_enc=self.cfg['share_feature'], dtype=self.cfg['dtype'], device=device)
        self.cfg.pop('dtype')

    def forward(self, data):
        self.eval()
        with torch.no_grad():
            choice, index, prob = self.act(data)
            V = self.criticize(data)
        self.train()
        return choice, index, prob, V
    
    def act(self, data, idx=None):
        return self.actor(data['N_obs'], data['info'], deterministic=True, choice=idx)

    def sample(self, data, tau=1):
        return self.actor(data['N_obs'], data['info'], tau=tau)

    def criticize(self, data):
        if self.cfg['share_feature']:
            tgt, agt = self.actor(data['state'], data['info'], feature_only=True)
            return self.critic((tgt, agt), data['info'])
        else:
            return self.critic(data['state'], data['info'])

class VotePtrActorCritic(Module):
    default = {
        'actor_cfg': {
            'common_cfg': {
                'embed_dim': 64,
                'activation': F.relu
            },

            'encoder_cfg': {
                'target_dim': 4,
                'agent_dim': 5,
                'enc_type': 'enc',
                'layer_num': 1,
                'nhead': 4,
                'cross_attn_en': False
            },

            'decoder_cfg': {
                'layer_num': 1, 
                'enc_type': 'sa',
                'query_dim': 256, 
                'nhead': 4
            }
        },
        'critic_cfg': {
            'common_cfg': {
                'embed_dim': 64,
                'activation': F.relu
            },

            'encoder_cfg': {
                'target_dim': 4,
                'agent_dim': 5,
                'enc_type': 'enc',
                'layer_num': 1,
                'nhead': 4,
                'cross_attn_en': False
            },

            'decoder_cfg': {
                'layer_num': 1, 
            }
        },
        'voting': True,
        'share_feature': True,
        'dtype': torch.float32
    }

    def __init__(self, cfg=None, device=None) -> None:
        super().__init__()
        self.cfg = deepcopy(self.default)
        if cfg is not None:
            self.cfg.update(cfg)
        self.voting = self.cfg['voting']
        self.training = True
        self.actor = MaPtrActor(**self.cfg['actor_cfg'], dtype=self.cfg['dtype'], device=device)
        self.critic = MaCritic(**self.cfg['critic_cfg'], share_enc=self.cfg['share_feature'], dtype=self.cfg['dtype'], device=device)
        self.cfg.pop('dtype')

    def forward(self, data):
        self.eval()
        with torch.no_grad():
            choice, index, prob = self.act(data)
            V = self.criticize(data)
        self.train()
        return choice, index, prob, V
    
    def act(self, data):
        seq_enc, _, _, prob = self.actor(data['N_obs'], data['info'], deterministic=True)
        return self.vote(seq_enc, prob, data)

    def sample(self, data, tau=1):
        seq_enc, _, _, prob = self.actor(data['N_obs'], data['info'], tau=tau)
        return self.vote(seq_enc, prob, data)
    
    def vote(self, seq_enc, prob, data):
        agent_num = data['state']['agents'].shape[1]
        com_num = data['N_obs']['agents'].shape[1]
        target_num = data['state']['targets'].shape[1]
        if self.voting:
            choice_count = torch.zeros((agent_num, target_num + 1))
            for idx_a, arrange in enumerate(seq_enc): # the agent index and arrangement of each agent
                agt_list = list(range(com_num))
                for agt in arrange['tar']:
                   agt_list.remove(agt)
                arrange['tar'] += agt_list # the agents don't have a target
                arrange = decode(arrange)
                for idx_t, agents in enumerate(arrange): # the target index and agent arranged to this target
                    for agent in agents:
                        agent_global_idx = data['info']['topk'][idx_a][agent]
                        if idx_a in data['info']['topk'][agent_global_idx]:
                            choice_count[agent_global_idx][idx_t] += 1
            choice_count = torch.roll(choice_count, 1, 1)
            index = torch.argmax(choice_count, dim=1)
        else:
            index = torch.zeros((agent_num, 1), dtype=int)
            for idx_a, arrange in enumerate(seq_enc): # the agent index and arrangement of each agent
                arrange = decode(arrange)
                for idx_t, agents in enumerate(arrange[:-1]): # the target index and agent arranged to this target
                    for agent in agents:
                        if agent == 0:
                            index[idx_a] = idx_t + 1
        choice = torch.zeros((agent_num, target_num + 1))
        for agt, idx in enumerate(index):
            choice[agt][idx] = 1
        return choice, index, torch.exp(torch.log(prob).sum(dim=-1))

    def criticize(self, data):
        if self.cfg['share_feature']:
            tgt, agt = self.actor(data['state'], data['info'], feature_only=True)
            return self.critic((tgt, agt), data['info'])
        else:
            return self.critic(data['state'], data['info'])
        
class MaBuffer():
    def __init__(self, buffer_size=10000, gamma=0.99, device=None, dtype=None) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.buffer_size = buffer_size
        self.gamma = gamma 
        self.reset()
    
    def store(self, s, a, r, p, v, d):
        if self.ptr >= self.buffer_size:
            print('buffer out of range ' + str(self.buffer_size))
            return
        self.state['N_obs']['agents'] = torch.cat([self.state['N_obs']['agents'], s['N_obs']['agents']], dim=0) if self.state['N_obs']['agents'] is not None else s['N_obs']['agents']
        self.state['N_obs']['targets'] = torch.cat([self.state['N_obs']['targets'], s['N_obs']['targets']], dim=0) if self.state['N_obs']['targets'] is not None else s['N_obs']['targets']
        self.a = torch.cat([self.a, a.flatten()]) if self.a is not None else a.flatten()
        self.split.append(len(self.a))
        self.rewards[self.ptr] = r
        self.log_p = torch.cat([self.log_p, torch.log(p.flatten())]) if self.log_p is not None else torch.log(p.flatten())
        self.values[self.ptr] = v.flatten()
        self.ptr += 1
        if d or self.ptr >= self.buffer_size:
            self.end_episode()
    
    def end_episode(self):
        q = self.rewards[self.start:(self.ptr-1)] + self.gamma * self.values[(self.start+1):self.ptr]
        self.adv[self.start:(self.ptr-1)] = q - self.values[self.start:(self.ptr-1)]
        self.adv[self.ptr-1] = self.rewards[self.ptr-1] - self.values[self.ptr-1]
        self.start = self.ptr

    def reset(self):
        self.state = {'N_obs': {'agents': None, 'targets': None}, 
                'info': {'agt_key_padding_mask': None, 'tgt_key_padding_mask': None}}
        self.a = None
        self.split = [0]
        self.rewards = torch.zeros((self.buffer_size), **self.factory_kwargs)
        self.log_p = None
        self.values = torch.zeros((self.buffer_size), **self.factory_kwargs)
        self.adv = torch.zeros((self.buffer_size), **self.factory_kwargs)
        self.ptr = 0
        self.start = 0

    def get(self, shuffle=False):
        expand_adv = torch.zeros_like(self.log_p)
        for idx, split in enumerate(zip(self.split[:-1], self.split[1:])):
            start, end = split
            expand_adv[start:end] = self.adv[idx].detach().item()
        
        data_idx = list(range(len(self.a)))
        if shuffle:
            random.shuffle(data_idx)
            self.state['N_obs']['agents'] = self.state['N_obs']['agents'][data_idx]
            self.state['N_obs']['targets'] = self.state['N_obs']['targets'][data_idx]
        return {
                's': self.state, 
                'a': self.a.unsqueeze(-1).unsqueeze(-1)[data_idx], 
                'split': self.split, 
                'log_p': self.log_p[data_idx], 
                'adv': self.adv, 
                'expand_adv': expand_adv[data_idx]
                }

def load_model(checkpoint):
    cfg = os.path.join('/'.join(checkpoint.split('/')[:-1]), 'config.json')
    with open(cfg, 'r') as f:
        cfg = json.load(f)
    
    ac_cfg = cfg['ac']

    ac = GNNAC(**ac_cfg)
    checkpoint = torch.load(checkpoint, map_location='cpu')
    ac.load_state_dict(checkpoint['model'])  
    ac.eval()
    ac.tau = cfg['anneal_final'] + (1 - cfg['anneal_final']) * (1 - checkpoint['epoch'] / cfg['epochs'])
    return ac

if __name__=="__main__":
    from torch.nn.utils.rnn import pad_sequence
    from torch_geometric.utils import unbatch, from_networkx
    from torch_geometric.loader import DataLoader
    from env import multiField
    import numpy as np
    import time
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # actor = MaPtrActor(device=device)
    # critic = MaCritic(device=device)
    # ac = MaPtrActorCritic(device=device)

    enc = MaGNNEncoder(device=device)
    ac = GNNAC(device=device, critic_list=['s', 't', 'c'], single_step=True)
    # pass
    bsz = 4
    # bsz_data = {'agents': [], 'targets': []}
    # info={'agt_key_padding_mask': [], 'tgt_key_padding_mask': [], 'num_tgt': torch.tensor([])}
    
    bsz_data = {'graph': [], 'vehicles': []}
    info = {'veh_key_padding_mask': [], 'num_veh': torch.tensor([])}
    fields = []
    car_cfgs = []

    tic = time.time()
    for idx in range(bsz):
    #     env = envGenerator(target_num=(idx+1)*3, agent_num=(idx + 1) * 3)
    #     data = env.encode()
    #     bsz_data['agents'].append(data['agents'])
    #     bsz_data['targets'].append(data['targets'])
    #     info['num_tgt'] = torch.cat([info['num_tgt'], torch.tensor([data['targets'].shape[0]])])
        field = multiField([1, 1], width = (600, 600), working_width=24) 
        fields.append(field)

        pygdata = from_networkx(field.working_graph, 
                                group_node_attrs=['embed'],
                                group_edge_attrs=['embed'])
        
        # bsz_data['vehicles'].append([
        #     {'vv': np.random.uniform(2, 4, 1), 
        #      'tt': 0, 
        #      'cw': np.random.uniform(0.005, 0.01, 1), 
        #      'cv': np.random.uniform(0.002, 0.005, 1),
        #      'vw': np.random.uniform(1, 3, 1), 
        #      'min_R': 7.15}
        #     for _ in range(idx+1)
        # ])
        bsz_data['vehicles'].append(torch.Tensor(
            np.array([
                np.random.uniform(1, 3.3, idx+1), # vw
                np.random.uniform(2, 25/3.6, idx+1), # vv
                np.random.uniform(0.007, 0.01, idx+1), #cw
                np.random.uniform(0.005, 0.008, idx+1), #cv
                np.random.uniform(0, 2, idx+1), #tt
            ]).T
        ))
        
        car_cfg = []
        for car in bsz_data['vehicles'][-1]:
            car_cfg.append({
                'vw': car[0].item(), 
                'vv': car[1].item(), 
                'cw': car[2].item(), 
                'cv': car[3].item(),
                'tt': 0, 
                'min_R': 7.15})
        car_cfgs.append(car_cfg)

        info['num_veh'] = torch.cat([info['num_veh'], torch.tensor([idx+1])])

        bsz_data['graph'].append(pygdata)

    # info['agt_key_padding_mask'] = [torch.zeros(len(x)) for x in bsz_data['agents']]
    # info['tgt_key_padding_mask'] = [torch.zeros(len(x)) for x in bsz_data['targets']]
    # bsz_data['agents'] = pad_sequence(bsz_data['agents'], batch_first=True, padding_value=0.).to(device)
    # bsz_data['targets'] = pad_sequence(bsz_data['targets'], batch_first=True, padding_value=0.).to(device)
    # info['agt_key_padding_mask'] = pad_sequence(info['agt_key_padding_mask'], batch_first=True, padding_value=1).bool().to(device)
    # info['tgt_key_padding_mask'] = pad_sequence(info['tgt_key_padding_mask'], batch_first=True, padding_value=1).bool().to(device)
    # info['num_tgt'] = info['num_tgt'].unsqueeze(1).to(device)
        
    print('Data generating time: ' + str(time.time() - tic))
    info['veh_key_padding_mask'] = [torch.zeros(len(x)) for x in bsz_data['vehicles']]
    info['veh_key_padding_mask'] = pad_sequence(info['veh_key_padding_mask'], batch_first=True, padding_value=1).bool().to(device)
    info['num_veh'] = info['num_veh'].view(-1, 1).to(device)
    
    bsz_data['vehicles'] = pad_sequence(bsz_data['vehicles'], batch_first=True).to(device)
    from torch_geometric.data import Batch

    tic = time.time()
    loader = DataLoader(bsz_data['graph'], batch_size=bsz)
    d = next(iter(loader)).to(device)
    print(time.time() - tic)
    tic = time.time()
    bsz_data['graph'] = Batch.from_data_list(bsz_data['graph']).to(device)
    print(time.time() - tic)
    
    # 0 is for depot
    chosen_idx = [
        [
            [1, 2, 3]
        ],
        [
            [1, 2, 3],
            [4, 5, 6, 7]
        ],
        [
            [4, 7],
            [3, 1, 6],
            [2, 5]
        ],
        [
            [3],
            [],
            [2, 4],
            []
        ]
    ]

    chosen_entry = [
        [
            [0, 1, 0]
        ],
        [
            [0, 1, 1],
            [1, 1, 1, 0]
        ],
        [
            [1, 1],
            [0, 0, 0],
            [1, 0]
        ],
        [
            [1],
            [],
            [0, 1],
            []
        ]
    ]

    # print(enc)
    # print(ac)
    # global_veh, veh, globel_nodes, nodes, node_key_padding_mask = enc(**bsz_data, veh_key_padding_mask=info['veh_key_padding_mask'])
    seq_enc, choice, index, entry, prob, dist1, value, value_mask = ac(bsz_data, info, chosen_idx=chosen_idx, chosen_entry=chosen_entry, force_chosen=True)
    # with torch.no_grad():
    #     seq_enc, choice, index, entry, prob, dist1, value, value_mask = ac(bsz_data, info, actor_grad=False)
    #     tic = time.time()
    #     seq_enc, choice, index, entry, prob, dist1, value, value_mask = ac(bsz_data, info, actor_grad=False)
    #     # seq_enc, choice, index, entry, prob, dist1, value_mask = ac(bsz_data, info, criticize=False)
    #     print(time.time() - tic)
    # seq_enc, choice, index, entry, prob, dist2, value, value_mask = ac(bsz_data, info, chosen_idx=index, chosen_entry=entry)
    # torch_kl = torch.distributions.kl.kl_divergence(dist1, dist2)
    # ac.eval()
    # seq_enc, choice, index, entry, prob, dist, value, value_mask = ac(bsz_data, info, deterministic=True)
    # seq_enc, choice, index, entry, prob, dist, value, value_mask = ac(bsz_data, info, chosen_idx=index, chosen_entry=entry)
    # seq_enc, choice, index, entry, prob, dist, value, value_mask = ac(bsz_data, info, deterministic=True)
    # loss = torch.log(prob).sum()
    # loss.backward(retain_graph=True)#
    # loss_c = value['default'].sum()
    # loss_c.backward()
    Ts = list(map(decode, seq_enc))
    # r = {'s': [], 't': [], 'c': []}
    # r_final = []
    # cost = {}
    # for key, val in value.items():
    #     cost.update({key: torch.zeros_like(val)})

    # for bidx, (T, field, car_cfg) in enumerate(zip(Ts, fields, car_cfgs)):
    #     ret = dense_fit(field.D_matrix, 
    #                 np.tile(field.ori, (len(car_cfg), 1, 1, 1)), 
    #                 np.tile(field.ori, (len(car_cfg), 1, 1, 1)),
    #                 car_cfg, 
    #                 field.line_length,
    #                 T)
    #     r_final.append(fit(field.D_matrix, 
    #                 np.tile(field.ori, (len(car_cfg), 1, 1, 1)), 
    #                 np.tile(field.ori, (len(car_cfg), 1, 1, 1)),
    #                 car_cfg, 
    #                 field.line_length,
    #                 T, 
    #                 tS_t=False,
    #                 type='all'))
    #     for key in r.keys():
    #         r[key].append(ret[key])
    # cur_cum_r = {}
    # for key in r.keys():
    #     r[key] = pad_sequence(r[key], batch_first=True)
    #     cum_key = torch.sum(r[key], dim=-1, keepdim=True)
    #     cur_cum_r.update({key: cum_key})

    # for key in cost.keys():
    #     if ac.single_step:
    #         cost[key][:, 0:1] = cur_cum_r[key]
    #     else: 
    #         length = r[key].shape[-1]
    #         cost[key][:, :length] = r[key].to(cost[key].device)
    
    # buffer = Buffer(0.99, 0.95, 'gae')
    # buffer.store(bsz_data, info, index, entry, cost, value, value_mask, prob, cur_cum_r, dist1)
    # data = buffer.get()

    pass