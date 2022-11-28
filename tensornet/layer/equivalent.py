import torch
from torch import nn
from typing import Dict, Callable, Optional, List
from tensornet.utils import way_combination, expand_to, multi_outer_product, find_distances
from tensornet.layer.radial import RadialLayer, ChebyshevPoly
from tensornet.layer.cutoff import CutoffLayer
import opt_einsum as oe


# input_tensors be like:
#   0: [n_batch, n_atoms, n_channel]
#   1: [n_batch, n_atoms, n_channel, n_dim]
#   2: [n_batch, n_atoms, n_channel, n_dim, n_dim]
#   .....
# coordinate: [n_batch, n_atoms, n_dim]


class TensorAggregateLayer(nn.Module):
    def __init__(self, 
                 radial_fn      : RadialLayer,
                 cutoff_fn      : CutoffLayer,
                 n_channel      : int,
                 max_out_way    : int=2, 
                 max_r_way      : int=2,
                 ) -> None:
        super().__init__()
        self.max_out_way = max_out_way
        self.max_r_way = max_r_way
        self.radial_fn = radial_fn
        self.cutoff_fn = cutoff_fn
        self.rbf_mixing_list = nn.ModuleList([
            nn.Linear(radial_fn.n_max, n_channel, bias=False) for i in range(max_r_way + 1)])

    def forward(self, 
                input_tensors : Dict[int, torch.Tensor],
                batch_data    : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = {way: None for way in range(self.max_out_way + 1)}
        neighbor = batch_data['neighbor']
        n_batch = neighbor.shape[0]
        device = neighbor.device
        idx_m = torch.arange(n_batch, device=device)[:, None, None]
        find_distances(batch_data)
        rij = batch_data['rij']
        dij = batch_data['dij']
        rbf_ij = self.radial_fn(dij) * self.cutoff_fn(dij)[..., None]  # [n_batch, n_atoms, n_neigh, n_rbf]
        input_tensor_dict = {}
        for in_way in input_tensors:
            input_tensor = input_tensors[in_way][idx_m, neighbor]
            mask = expand_to(batch_data['mask'], 4 + in_way)
            input_tensor_dict[in_way] = input_tensor.masked_fill(mask=mask, value=0.)

        filter_tensor_dict = {}
        for out_way, in_way, r_way in way_combination(range(self.max_out_way + 1), 
                                                      input_tensors.keys(), 
                                                      range(self.max_r_way + 1)):
            if r_way not in filter_tensor_dict:
                fn = self.rbf_mixing_list[r_way](rbf_ij)       # [n_batch, n_atoms, n_neigh, n_channel]
                fn = fn * input_tensor_dict[0]                 # [n_batch, n_atoms, n_neigh, n_channel]
                rij_tensor = multi_outer_product(rij, r_way)   # [n_batch, n_atoms, n_neigh, n_dim, n_dim, ...]
                filter_tensor = rij_tensor.unsqueeze(3) * expand_to(fn, n_dim=r_way + 4)
                filter_tensor_dict[r_way] = filter_tensor      # [n_batch, n_atoms, n_neigh, n_channel, n_dim, n_dim, ...]
            filter_tensor = filter_tensor_dict[r_way]          # [n_batch, n_atoms, n_neigh, n_channel, n_dim, n_dim, ...]
            input_tensor  = input_tensor_dict[in_way]

            # filter_tensor: [n_batch, n_atoms, n_neigh, n_channel, n_dim, n_dim, ...]   
            #                with  (r_way) n_dim
            # input_tensor:  [n_batch, n_atoms, n_neigh, n_channel, n_dim, n_dim, ...]  
            #                with (in_way) n_dim

            coupling_way = (in_way + r_way - out_way) // 2

            # # method 1 
            n_way = in_way + r_way - coupling_way + 4
            input_tensor  = expand_to(input_tensor, n_way, dim=-1)
            filter_tensor = expand_to(filter_tensor, n_way, dim=4)

            # input_tensor:  [n_batch, n_atoms, n_neigh, n_channel, n_dim, n_dim, ...,     1] 
            # filter_tensor: [n_batch, n_atoms, n_neigh, n_channel,     1,     1, ..., n_dim]  
            # with (in_way + r_way - coupling_way) dim after n_channel
            # We should sum up 2 (n_neigh) and (coupling_way) n_dim
            sum_axis = [2] + [i for i in range(in_way - coupling_way + 4, in_way + 4)]
            output_tensor = torch.sum(input_tensor * filter_tensor, dim=sum_axis)

            # method 2
            # Why opt_einsum slower?
            # input_indices = [i for i in range(4, 4 + in_way - coupling_way)]
            # coupling_indices = [i for i in range(4 + in_way - coupling_way, 4 + in_way)]
            # filter_indices = [i for i in range(4 + in_way, 4 + in_way + r_way - coupling_way)]

            # input_subscripts  = [0, 1, 2, 3] + input_indices + coupling_indices
            # filter_subscripts = [0, 1, 2, 3] + coupling_indices + filter_indices
            # output_subscripts = [0, 1, 2, 3] + input_indices + filter_indices
            # output_tensor = oe.contract(input_tensor, input_subscripts, filter_tensor, filter_subscripts, output_subscripts)

            if output_tensors[out_way] is None:
                output_tensors[out_way]  = output_tensor
            else:
                output_tensors[out_way] += output_tensor 
            # if out_way == 1:
            #     print(f"rway\n:{r_way}\nin_way:{in_way}\nout_way:\n{out_way}\n"
            #           f"filter:\n{filter_tensor}\ninput:\n{input_tensor}\noutput:\n{output_tensor}")
            
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
            input_tensor = torch.transpose(input_tensors[way], 2, -1)
            output_tensor = self.linear_interact_list[way](input_tensor)
            output_tensors[way] = torch.transpose(output_tensor, 2, -1)
        return output_tensors


# TODO: cat different way together and use Linear layer to got factor of every channel
class NonLinearLayer(nn.Module):
    def __init__(self, 
                 activate_fn : Callable,
                 max_in_way  : int,
                 ) -> None:
        super().__init__()
        self.activate_fn = activate_fn
        # TODO: how to initialize parameters?
        self.weights = nn.Parameter(torch.ones(max_in_way + 1))
        self.bias = nn.Parameter(torch.ones(max_in_way + 1))

    def forward(self,
                input_tensors: torch.Tensor,
                ) -> torch.Tensor:
        output_tensors = {}
        for way in input_tensors:
            if way == 0:
                output_tensor = self.activate_fn(self.weights[way] * input_tensors[way] + self.bias[way])
            else:
                norm = torch.linalg.norm(input_tensors[way], dim=tuple(range(3, 3 + way)), keepdim=True)
                #print(f'way:\n{way}\norigin:\n{input_tensors[way]}norm:{norm}')
                factor = self.activate_fn(self.weights[way] * norm + self.bias[way])
                #print(factor)s
                output_tensor = factor * input_tensors[way]
            output_tensors[way] = output_tensor
        return output_tensors


class SOnEquivalentLayer(nn.Module):
    def __init__(self,
                 activate_fn    : Callable,
                 radial_fn      : RadialLayer,
                 cutoff_fn      : CutoffLayer,
                 max_r_way      : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 ) -> None:
        super().__init__()
        self.tensor_aggregate = TensorAggregateLayer(radial_fn=radial_fn,
                                                     cutoff_fn=cutoff_fn,
                                                     n_channel=input_dim,
                                                     max_out_way=max_out_way, 
                                                     max_r_way=max_r_way)
        # input for SelfInteractionLayer and NonLinearLayer is the output of TensorAggregateLayer
        # so the max_in_way should equal to max_out_way of TensorAggregateLayer
        self.self_interact = SelfInteractionLayer(input_dim=input_dim, 
                                                  max_in_way=max_out_way, 
                                                  output_dim=output_dim)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_in_way=max_out_way)

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                batch_data    : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = self.tensor_aggregate(input_tensors=input_tensors, 
                                               batch_data=batch_data)
        output_tensors = self.self_interact(output_tensors)
        output_tensors = self.non_linear(output_tensors)
        return output_tensors
