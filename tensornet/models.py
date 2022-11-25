import torch
from torch import nn
from typing import Callable, List, Dict, Optional
from tensornet.layer.equivalent import SOnEquivalentLayer, SelfInteractionLayer
from tensornet.layer.embedding import EmbeddingLayer
from tensornet.layer.radius import RadiusFunction
from tensornet.layer.readout import ReadoutLayer
from tensornet.utils import expand_to, find_distances


def expand_para(para: int or List, n: int):
    assert isinstance(para, int) or isinstance(para, list)
    if isinstance(para, int):
        para = [para] * n
    if isinstance(para, list):
        assert len(para) == n
    return para    


class ANI(nn.Module):
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

    def forward(self, 
                batch_data : Dict[str, torch.Tensor],
                ) -> torch.Tensor:
        # TODO: or move this to data prepare?
        find_distances(batch_data)
        output_tensors = self.embedding_layer(batch_data=batch_data)
        for layer in self.linear_layers:
            output_tensors = self.activate_fn(layer(output_tensors))
        output_tensors = self.readout_layer(output_tensors).squeeze(2)
        symbol_mask = batch_data['atomic_number'] < 0.5
        output_tensors.masked_fill(mask=symbol_mask, value=torch.tensor(0.))
        return output_tensors
    
    
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
                 radius_fn_para  : Dict={},
                 ):
        super().__init__()
        self.embedding_layer = embedding_layer
        max_r_way = expand_para(max_r_way, n_layers)
        max_out_way = expand_para(max_out_way, n_layers)
        hidden_nodes = [embedding_layer.n_channel] + expand_para(output_dim, n_layers)
        self.son_equivalent_layers = nn.ModuleList(
            [SOnEquivalentLayer(activate_fn=activate_fn,
                                radius_fn=radius_fn,
                                radius_fn_para=radius_fn_para,
                                max_r_way=max_r_way[i],
                                max_out_way=max_out_way[i],
                                input_dim=hidden_nodes[i],
                                output_dim=hidden_nodes[i + 1]) for i in range(n_layers)])
        self.readout_layer = ReadoutLayer(n_dim=hidden_nodes[-1],
                                          target_way=target_way)

    def forward(self,
                batch_data : Dict[str, torch.Tensor],
                # coordinate    : torch.Tensor,
                # atomic_number : torch.Tensor,
                # neighbor      : torch.Tensor,
                # mask          : torch.Tensor,
                # cell          : Optional[torch.Tensor]=None,
                # offset        : Optional[torch.Tensor]=None,
                ) -> Dict[int, torch.Tensor]:
        # TODO: or move this to data prepare?
        find_distances(batch_data)
        output_tensors = {0: self.embedding_layer(batch_data=batch_data)}
        for layer in self.son_equivalent_layers:
            output_tensors = layer(input_tensors=output_tensors, 
                                   batch_data=batch_data)
        output_tensors = self.readout_layer(output_tensors,
                                            atomic_number=batch_data['atomic_number'])
        return output_tensors
