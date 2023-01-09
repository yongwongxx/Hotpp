# TODO 
# resnet
# Now segment_coo have some bug in getting second-order derivative
# https://github.com/rusty1s/pytorch_scatter/issues/299
# from torch_scatter import segment_coo

import torch
from torch import nn
from typing import Dict, Callable, Union
from .base import RadialLayer, CutoffLayer
from .activate import TensorActivateDict
from ..utils import find_distances, expand_to, way_combination, multi_outer_product, _scatter_add


# input_tensors be like:
#   0: [n_atoms, n_channel]
#   1: [n_atoms, n_channel, n_dim]
#   2: [n_atoms, n_channel, n_dim, n_dim]
#   .....
# coordinate: [n_atoms, n_dim]

__all__ = ["TensorLinear",
           "TensorAggregateLayer",
           "SelfInteractionLayer",
           "NonLinearLayer",
           "SOnEquivalentLayer",
           ]


class TensorLinear(nn.Module):
    def __init__(self,
                 input_dim   : int,
                 output_dim  : int,
                 bias        : bool=False,
                 ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, 
                input_tensor: torch.Tensor,   # [n_batch, n_channel, n_dim, n_dim, ...]
                ):
        way = len(input_tensor.shape) - 2
        if way == 0:
            output_tensor = self.linear(input_tensor)
        else:
            input_tensor = torch.transpose(input_tensor, 1, -1)
            output_tensor = self.linear(input_tensor)
            output_tensor = torch.transpose(output_tensor, 1, -1)
        return output_tensor


class TensorAggregateLayer(nn.Module):
    def __init__(self, 
                 radial_fn      : RadialLayer,
                 cutoff_fn      : CutoffLayer,
                 n_channel      : int,
                 max_out_way    : int=2, 
                 max_r_way      : int=2,
                 norm_factor    : float=1.,
                 ) -> None:
        super().__init__()
        self.max_out_way = max_out_way
        self.max_r_way = max_r_way
        self.radial_fn = radial_fn
        self.cutoff_fn = cutoff_fn
        self.rbf_mixing_list = nn.ModuleList([
            nn.Linear(radial_fn.n_max, n_channel, bias=False) for i in range(max_r_way + 1)])
        self.register_buffer("norm_factor", torch.tensor(norm_factor).float())

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                batch_data    : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:

        output_tensors = {way: None for way in range(self.max_out_way + 1)}
        idx_i, idx_j = batch_data['edge_index']
        n_atoms = batch_data['atomic_number'].shape[0]
        _, dij, uij = find_distances(batch_data)
        rbf_ij = self.radial_fn(dij) * self.cutoff_fn(dij)[..., None]  # [n_edge, n_rbf]

        filter_tensor_dict = {}
        for out_way, in_way, r_way in way_combination(range(self.max_out_way + 1), 
                                                      input_tensors.keys(), 
                                                      range(self.max_r_way + 1)):
            if r_way not in filter_tensor_dict:
                fn = self.rbf_mixing_list[r_way](rbf_ij)          # [n_edge, n_channel]
                # TODO: WHY!!!!!!!!!! CAO!
                # fn = fn * input_tensor_dict[0]                  # [n_edge, n_channel]
                moment_tensor = multi_outer_product(uij, r_way)   # [n_edge, n_dim, ...]
                filter_tensor = moment_tensor.unsqueeze(1) * expand_to(fn, n_dim=r_way + 2)
                filter_tensor_dict[r_way] = filter_tensor         # [n_edge, n_channel, n_dim, n_dim, ...]
            filter_tensor = filter_tensor_dict[r_way]             # [n_edge, n_channel, n_dim, n_dim, ...]
            input_tensor = input_tensors[in_way][idx_j]           # [n_edge, n_channel, n_dim, n_dim, ...]
            coupling_way = (in_way + r_way - out_way) // 2
            # # method 1 
            n_way = in_way + r_way - coupling_way + 2
            input_tensor  = expand_to(input_tensor, n_way, dim=-1)
            filter_tensor = expand_to(filter_tensor, n_way, dim=2)
            output_tensor = input_tensor * filter_tensor
            # input_tensor:  [n_edge, n_channel, n_dim, n_dim, ...,     1] 
            # filter_tensor: [n_edge, n_channel,     1,     1, ..., n_dim]  
            # with (in_way + r_way - coupling_way) dim after n_channel
            # We should sum up (coupling_way) n_dim
            if coupling_way > 0:
                sum_axis = [i for i in range(in_way - coupling_way + 2, in_way + 2)]
                output_tensor = torch.sum(output_tensor, dim=sum_axis)
            output_tensor = _scatter_add(output_tensor, idx_i, dim_size=n_atoms) / self.norm_factor
            # output_tensor = segment_coo(output_tensor, idx_i, dim_size=batch_data.num_nodes, reduce="sum")

            if output_tensors[out_way] is None:
                output_tensors[out_way] = output_tensor
            else:
                output_tensors[out_way] += output_tensor
        return output_tensors


class SelfInteractionLayer(nn.Module):
    def __init__(self, 
                 input_dim  : int,
                 max_in_way : int,
                 output_dim : int=10,
                 ) -> None:
        super().__init__()
        # only the way 0 can have bias
        self.linear_interact_list = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        for way in range(1, max_in_way + 1):
            self.linear_interact_list.append(nn.Linear(input_dim, output_dim, bias=False))

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = {}
        for way in input_tensors:
            # swap channel axis and the last dim axis
            input_tensor = torch.transpose(input_tensors[way], 1, -1)
            output_tensor = self.linear_interact_list[way](input_tensor)
            output_tensors[way] = torch.transpose(output_tensor, 1, -1)
        return output_tensors


# TODO: cat different way together and use Linear layer to got factor of every channel
class NonLinearLayer(nn.Module):
    def __init__(self, 
                 max_in_way  : int,
                 input_dim   : int,
                 activate_fn : str='jilu',
                 ) -> None:
        super().__init__()
        self.activate_list = nn.ModuleList([TensorActivateDict[activate_fn](input_dim) 
                                            for _ in range(max_in_way + 1)])

    def forward(self,
                input_tensors: torch.Tensor,
                ) -> torch.Tensor:
        output_tensors = {}
        for way in input_tensors: 
            output_tensors[way] = self.activate_list[way](input_tensors[way])
        return output_tensors


class SOnEquivalentLayer(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 cutoff_fn      : CutoffLayer,
                 max_r_way      : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 norm_factor    : float=1.0,
                 activate_fn    : str='jilu',
                 ) -> None:
        super().__init__()
        self.tensor_aggregate = TensorAggregateLayer(radial_fn=radial_fn,
                                                     cutoff_fn=cutoff_fn,
                                                     n_channel=input_dim,
                                                     max_out_way=max_out_way, 
                                                     max_r_way=max_r_way,
                                                     norm_factor=norm_factor,)
        # input for SelfInteractionLayer and NonLinearLayer is the output of TensorAggregateLayer
        # so the max_in_way should equal to max_out_way of TensorAggregateLayer
        self.self_interact = SelfInteractionLayer(input_dim=input_dim, 
                                                  max_in_way=max_out_way, 
                                                  output_dim=output_dim)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_in_way=max_out_way,
                                         input_dim=output_dim)

    def forward(self,
                batch_data : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        batch_data['node_attr'] = self.propagate(batch_data['node_attr'], batch_data)
        return batch_data

    # TODO: sparse version
    def message_and_aggregate(self,
                              input_tensors : Dict[int, torch.Tensor],
                              batch_data    : Dict[int, torch.Tensor],
                              ) -> Dict[int, torch.Tensor]:
        output_tensors =  self.tensor_aggregate(input_tensors=input_tensors, 
                                                batch_data=batch_data)
        # resnet
        for r_way in input_tensors.keys():
            output_tensors[r_way] += input_tensors[r_way]
        return output_tensors

    def update(self,
               input_tensors : Dict[int, torch.Tensor],
               ) -> Dict[int, torch.Tensor]:
        output_tensors = self.self_interact(input_tensors)
        output_tensors = self.non_linear(output_tensors)
        # resnet
        for r_way in input_tensors.keys():
            output_tensors[r_way] += input_tensors[r_way]
        return output_tensors

    def propagate(self,
                  input_tensors : Dict[int, torch.Tensor],
                  batch_data : Dict[int, torch.Tensor],
                  ) -> Dict[int, torch.Tensor]:
        output_tensors = self.message_and_aggregate(input_tensors, batch_data)
        output_tensors = self.update(output_tensors)
        return output_tensors
