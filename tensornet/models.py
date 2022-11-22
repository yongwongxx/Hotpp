import torch
from torch import nn
from typing import Callable, List, Dict, Optional
from tensornet.layer.equivalent import SOnEquivalentLayer, SelfInteractionLayer
from tensornet.layer.embedding import EmbeddingLayer
from tensornet.layer.radius import RadiusFunction


def expand_para(para: int or List, n: int):
    assert isinstance(para, int) or isinstance(para, list)
    if isinstance(para, int):
        para = [para] * n
    if isinstance(para, list):
        assert len(para) == n
    return para    


class TensorMessagePassingNet(nn.Module):

    def __init__(self,
                 embedding_layer : EmbeddingLayer,
                 radius_fn       : RadiusFunction,
                 n_layers        : int,
                 max_r_way       : int or List,
                 max_out_way     : int or List,
                 output_dim      : int or List,
                 activate_fn     : Callable,
                 target_way      : List[int]=[0],
                 ):
        super().__init__()
        self.embedding_layer = embedding_layer
        max_r_way = expand_para(max_r_way, n_layers)
        max_out_way = expand_para(max_out_way, n_layers)
        hidden_nodes = [embedding_layer.n_channel] + expand_para(output_dim, n_layers)
        self.son_equivalent_layers = nn.ModuleList(
            [SOnEquivalentLayer(activate_fn=activate_fn,
                                radius_fn=radius_fn,
                                max_r_way=max_r_way[i],
                                max_out_way=max_out_way[i],
                                input_dim=hidden_nodes[i],
                                output_dim=hidden_nodes[i + 1]) for i in range(n_layers)])
        self.readout_layer = SelfInteractionLayer(input_dim=hidden_nodes[-1],
                                                  max_in_way=max_out_way[-1],
                                                  output_dim=1)
        self.target_way = target_way

    def forward(self,
                coordinate    : torch.Tensor,
                atomic_number : torch.Tensor,
                neighbor      : torch.Tensor,
                mask          : torch.Tensor,
                cell          : Optional[torch.Tensor]=None,
                offset        : Optional[torch.Tensor]=None,
                ) -> Dict[int, torch.Tensor]:

        output_tensors = {0: self.embedding_layer(coordinate=coordinate,
                                                  atomic_number=atomic_number,
                                                  neighbor=neighbor,
                                                  mask=mask,
                                                  cell=cell,
                                                  offset=offset)}
        for layer in self.son_equivalent_layers:
            output_tensors = layer(input_tensors=output_tensors, 
                                   coordinate=coordinate, 
                                   neighbor=neighbor,
                                   mask=mask,
                                   cell=cell)
        output_tensors = self.readout_layer(output_tensors)
        result = {way: output_tensors[way] for way in self.target_way}
        return result
