from typing import Callable, List, Dict, Optional
import torch
from torch import nn
from .base import AtomicModule
from ..layer import SOnEquivalentLayer, EmbeddingLayer, RadialLayer, ReadoutLayer, CutoffLayer
from ..utils import expand_para, find_distances


class MiaoNet(AtomicModule):
    """
    Miao nei ga
    duo xi da miao nei
    """
    def __init__(self,
                 embedding_layer : EmbeddingLayer,
                 radial_fn       : RadialLayer,
                 cutoff_fn       : CutoffLayer,
                 n_layers        : int,
                 max_r_way       : int or List,
                 max_out_way     : int or List,
                 output_dim      : int or List,
                 activate_fn     : Callable,
                 target_way      : List[int]=[0],
                 mean            : float=0.,
                 std             : float=1.,
                 norm_factor     : float=1.,
                 ):
        super().__init__(mean=mean, std=std)
        self.embedding_layer = embedding_layer
        max_r_way = expand_para(max_r_way, n_layers)
        max_out_way = expand_para(max_out_way, n_layers)
        hidden_nodes = [embedding_layer.n_channel] + expand_para(output_dim, n_layers)
        self.son_equivalent_layers = nn.ModuleList(
            [SOnEquivalentLayer(activate_fn=activate_fn,
                                radial_fn=radial_fn,
                                cutoff_fn=cutoff_fn,
                                max_r_way=max_r_way[i],
                                max_out_way=max_out_way[i],
                                input_dim=hidden_nodes[i],
                                output_dim=hidden_nodes[i + 1],
                                norm_factor=norm_factor) for i in range(n_layers)])
        self.readout_layer = ReadoutLayer(n_dim=hidden_nodes[-1],
                                          target_way=target_way, 
                                          activate_fn=activate_fn)

    def calculate(self,
                  batch_data : Dict[str, torch.Tensor],
                  ) -> torch.Tensor:
        find_distances(batch_data)
        batch_data['node_attr'] = {0: self.embedding_layer(batch_data=batch_data)}
        for layer in self.son_equivalent_layers:
            batch_data = layer(batch_data)
        output_tensors = self.readout_layer(batch_data)
        return output_tensors
