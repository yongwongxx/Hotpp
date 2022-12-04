from typing import List, Callable, Dict
import torch
from torch import nn
from .base import AtomicModule
from ..layer import EmbeddingLayer
from ..utils import expand_para, find_distances

class ANI(AtomicModule):
    def __init__(self,
                 embedding_layer : EmbeddingLayer,
                 n_layers        : int,
                 output_dim      : int or List,
                 activate_fn     : Callable,
                 ) -> None:
        super().__init__()
        self.embedding_layer = embedding_layer
        hidden_nodes = [embedding_layer.n_channel] + expand_para(output_dim, n_layers)
        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]) for i in range(n_layers)])
        self.readout_layer = nn.Linear(hidden_nodes[-1], 1)
        self.activate_fn = activate_fn

    def get_site_energy(self, 
                        batch_data : Dict[str, torch.Tensor],
                        ) -> torch.Tensor:
        find_distances(batch_data)
        output_tensors = self.embedding_layer(batch_data=batch_data)
        for layer in self.linear_layers:
            output_tensors = self.activate_fn(layer(output_tensors))
        output_tensors = self.readout_layer(output_tensors).squeeze(1)
        return output_tensors