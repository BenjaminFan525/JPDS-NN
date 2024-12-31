import torch
import torch.nn as nn
from torch.nn import Module
from typing import Optional, Tuple
import torch.nn.functional as F
import math
from copy import deepcopy

activations = {
    'relu': F.relu,
    'gelu': F.gelu,
    'tanh': F.tanh,
    'sigmoid': F.sigmoid
}

class FeatureBlock(Module):
    def __init__(self, enc_type='sa', layer_num=1, embed_dim=64, nhead=4, activation=F.relu, dropout=0., device=None, dtype=None) -> None:
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        if isinstance(activation, str):
            activation = activations[activation]
        super().__init__()
        self.activation = activation
        self.embed_dim = embed_dim

        self.enc_type = enc_type
        if enc_type == 'enc':
            layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, dropout=dropout, nhead=nhead, dim_feedforward=self.embed_dim, batch_first=True, activation=self.activation, **self.factory_kwargs)
        elif enc_type == 'mlp':
            layer = nn.Linear(self.embed_dim, self.embed_dim, bias=False, **self.factory_kwargs)
        elif enc_type == 'sa':
            layer = nn.MultiheadAttention(self.embed_dim, nhead, dropout=dropout, batch_first=True, **self.factory_kwargs)

        self.mods = nn.ModuleList([deepcopy(layer) for _ in range(layer_num)])

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        for mod in self.mods:
            if self.enc_type == 'enc':
                x = self.activation(mod(x, src_key_padding_mask=key_padding_mask, src_mask=attn_mask))
            elif self.enc_type == 'mlp':
                x = self.activation(mod(x))
            elif self.enc_type == 'sa':
                x = self.activation(mod(x, x, x, key_padding_mask=key_padding_mask)[0])
        return x

class Embedding_layer(Module):
    def __init__(self, input_dim: int, embed_dim: int, embed_layer: int, 
                 bias=True, activation=F.relu, device='cpu', dtype=torch.float32) -> None:
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.activation = activation
        self.embedding = nn.ModuleList([nn.Linear(input_dim, embed_dim, bias=bias, **self.factory_kwargs)] + \
            [nn.Linear(embed_dim, embed_dim, bias=bias, **self.factory_kwargs) for _ in range(embed_layer - 1)])
    
    def forward(self, x):
        for mod in self.embedding[:-1]:
            x = self.activation(mod(x))
        return self.embedding[-1](x)
    
class PositionalEncoding(Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, batch_first=False, device='cpu', dtype=torch.float32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, device=device, dtype=dtype)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.batch_first = batch_first
        if batch_first:
            pe = torch.transpose(pe, 0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[0, :x.size(1)] if self.batch_first else self.pe[:x.size(0)]
        return self.dropout(x)