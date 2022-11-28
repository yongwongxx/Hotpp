import torch
from torch import nn
from typing import Callable, List, Dict, Optional
from tensornet.layer.equivalent import SOnEquivalentLayer, SelfInteractionLayer
from tensornet.layer.embedding import EmbeddingLayer
from tensornet.layer.radial import RadialLayer
from tensornet.layer.readout import ReadoutLayer
from tensornet.layer.cutoff import CutoffLayer
from tensornet.utils import expand_to, find_distances


def expand_para(para: int or List, n: int):
    assert isinstance(para, int) or isinstance(para, list)
    if isinstance(para, int):
        para = [para] * n
    if isinstance(para, list):
        assert len(para) == n
    return para    


class AtomicModule(nn.Module):
    def forward(self, 
                batch_data  : Dict[str, torch.Tensor],
                properties  : List[str]=['energy'],
                ) -> Dict[str, torch.Tensor]:
        if 'forces' in properties:
            batch_data['coordinate'].requires_grad_()
        site_energy = self.get_site_energy(batch_data)
        if 'energy' in properties:
            batch_data['energy_p'] = site_energy.sum(dim=1)
        if 'forces' in properties:
            batch_data['forces_p'] = -torch.autograd.grad(site_energy.sum(),
                                                          batch_data['coordinate'],
                                                          create_graph=True,
                                                          retain_graph=True)[0]
        return batch_data

    def get_site_energy(self):
        raise NotImplementedError(f"{self.__class__.__name__} must have 'get_energy'!")



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
        # TODO: or move this to data prepare?
        find_distances(batch_data)
        output_tensors = self.embedding_layer(batch_data=batch_data)
        for layer in self.linear_layers:
            output_tensors = self.activate_fn(layer(output_tensors))
        output_tensors = self.readout_layer(output_tensors).squeeze(2)
        symbol_mask = batch_data['atomic_number'] < 0.5
        output_tensors.masked_fill_(mask=symbol_mask, value=0.)
        return output_tensors


class TensorMessagePassingNet(AtomicModule):

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
                 ):
        super().__init__()
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
                                output_dim=hidden_nodes[i + 1]) for i in range(n_layers)])
        self.readout_layer = ReadoutLayer(n_dim=hidden_nodes[-1],
                                          target_way=target_way)

    def get_site_energy(self,
                        batch_data : Dict[str, torch.Tensor],
                        ) -> Dict[int, torch.Tensor]:
        # TODO: or move this to data prepare?
        find_distances(batch_data)
        output_tensors = {0: self.embedding_layer(batch_data=batch_data)}
        for layer in self.son_equivalent_layers:
            output_tensors = layer(input_tensors=output_tensors, 
                                   batch_data=batch_data)
        output_tensors = self.readout_layer(output_tensors,
                                            atomic_number=batch_data['atomic_number'])
        return output_tensors[0]
