import torch
from torch import nn
from typing import Dict, Callable, Optional, List
from tensornet.utils import way_combination, expand_to, multi_outer_product, find_distances
from tensornet.layer.radius import RadiusFunction, ChebyshevPoly

##############  debug  #################
import numpy as np
R = torch.tensor([
    [-0.85771824,  0.45413047, -0.24100817],
    [-0.1945646,  -0.720634,   -0.66545567],
    [-0.47588238, -0.52388181,  0.70645864]])
#########################################

class TensorAggregateLayer(nn.Module):
    def __init__(self, 
                 radius_fn   : RadiusFunction,
                 max_out_way : int=2, 
                 max_r_way   : int=2,
                 ) -> None:
        super().__init__()
        self.max_out_way = max_out_way
        self.max_r_way = max_r_way
        self.radius_fn_list = nn.ModuleList([radius_fn for _ in range(max_r_way + 1)])

    def forward(self, 
                input_tensors : Dict[int, torch.Tensor],
                coordinate    : torch.Tensor,                   
                neighbor      : torch.Tensor,
                mask          : torch.Tensor,
                cell          : Optional[torch.Tensor]=None,
                offset        : Optional[torch.Tensor]=None,
                ) -> Dict[int, torch.Tensor]:
        # input_tensors:
        #   0: [n_batch, n_atoms, n_channel]
        #   1: [n_batch, n_atoms, n_channel, n_dim]
        #   2: [n_batch, n_atoms, n_channel, n_dim, n_dim]
        #   .....
        # coordinate: [n_batch, n_atoms, n_dim]
        output_tensors = {way: None for way in range(self.max_out_way + 1)}

        for out_way, in_way, r_way in way_combination(range(self.max_out_way + 1), 
                                                      input_tensors.keys(), 
                                                      range(self.max_r_way + 1)):
            coupling_way = (in_way + r_way - out_way) // 2
            rij = find_distances(coordinate=coordinate,
                                 neighbor=neighbor,
                                 mask=mask,
                                 cell=cell,
                                 offset=offset)
            dij = torch.linalg.norm(rij, dim=-1)
            fn = self.radius_fn_list[r_way](dij)    # [n_batch, n_atoms, n_neigh]
            filter_tensor = multi_outer_product(rij, r_way) * expand_to(fn, n_dim=r_way + 3)

            ###################################################################
            # print("Filter")
            # print(out_way, in_way, r_way)
            # if r_way == 0:
            #     print(filter_tensor[0] - filter_tensor[1])
            # if r_way == 1:
            #     R1 = filter_tensor[0, :, 0]
            #     R2 = filter_tensor[1, :, 0]
            #     print((R1 @ R.T - R2) / R1)
            # if r_way == 2:
            #     T1 = filter_tensor[0, :, 0]
            #     T2 = filter_tensor[1, :, 0]
            #     print((T2 - torch.matmul(torch.matmul(R, T1), R.T)) / T2)
            #################################################################
            # print("In")
            # print(out_way, in_way, r_way)
            # if in_way == 0:
            #     print(input_tensors[in_way][0] - input_tensors[in_way][1])
            # if in_way == 1:
            #     R1 = input_tensors[in_way][0, :, 0]
            #     R2 = input_tensors[in_way][1, :, 0]
            #     print((R1 @ R.T - R2) / R1)
            # if in_way == 2:
            #     T1 = input_tensors[in_way][0, :, 0]
            #     T2 = input_tensors[in_way][1, :, 0]
            #     print((T2 - torch.matmul(torch.matmul(R, T1), R.T)) / T2)
            ##################################################################

            # filter_tensor: 
            #   [n_batch, n_atoms, n_neigh, n_dim, n_dim, ...]    with  (r_way) n_dim
            # input_tensors[in_way]:
            #   [n_batch, n_atoms, n_channel, n_dim, n_dim, ...]  with (in_way) n_dim
            # TODO: Will use torch.einsum faster?
            n_way = in_way + r_way - coupling_way + 4
            input_tensor  = expand_to(input_tensors[in_way][:, :, :, None, ...], n_way, dim=-1)
            filter_tensor = expand_to(filter_tensor[:, :, None, ...], n_way, dim=4)

            # [n_batch, n_atoms, n_channel, n_neigh, n_dim, n_dim, ...]  with (in_way + r_way - coupling_way) n_dim
            # We should sum up 3 (n_neigh) and (coupling_way) n_dim
            sum_axis = [3] + [i for i in range(in_way - coupling_way + 4, in_way + 4)]
            output_tensor = torch.sum(input_tensor * filter_tensor, dim=sum_axis)
            if output_tensors[out_way] is None:
                output_tensors[out_way]  = output_tensor
            else:
                output_tensors[out_way] += output_tensor 

            ###################################################################
            # print("Aggerate")
            # print(out_way, in_way, r_way)
            # if out_way == 0:
            #     print(output_tensor[0] - output_tensor[1])
            # if out_way == 1:
            #     R1 = output_tensor[0, :, 0]
            #     R2 = output_tensor[1, :, 0]
            #     print((R1 @ R.T - R2) / R1)
            # if out_way == 2:
            #     T1 = output_tensor[0, :, 0]
            #     T2 = output_tensor[1, :, 0]
            #     print((T2 - torch.matmul(torch.matmul(R, T1), R.T)) / T2)
            ##################################################################

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
        # input_tensors:
        #   0: [n_batch, n_atoms, n_channel]
        #   1: [n_batch, n_atoms, n_channel, n_dim]
        #   2: [n_batch, n_atoms, n_channel, n_dim, n_dim]
        #   .....
        output_tensors = {}
        for way in input_tensors:
            # swap channel axis and the last dim axis
            input_tensor = torch.transpose(input_tensors[way], 2, -1)
            output_tensor = self.linear_interact_list[way](input_tensor)
            output_tensors[way] = torch.transpose(output_tensor, 2, -1)
            ###################################################################
            # print("Self")
            # print(way)
            # if way == 0:
            #     print(output_tensors[way][0] - output_tensors[way][1])
            # if way == 1:
            #     R1 = output_tensors[way][0, :, 0]
            #     R2 = output_tensors[way][1, :, 0]
            #     print(R1 @ R.T - R2)
            # if way == 2:
            #     T1 = output_tensors[way][0, :, 0]
            #     T2 = output_tensors[way][1, :, 0]
            #     print(T2 - torch.matmul(torch.matmul(R, T1), R.T))
            ##################################################################
        return output_tensors


class NonLinearLayer(nn.Module):
    def __init__(self, 
                 activate_fn : Callable,
                 max_in_way  : int,
                 ) -> None:
        super().__init__()
        self.activate_fn = activate_fn
        # TODO: how to initialize parameters?
        self.weights = nn.Parameter(torch.ones(max_in_way + 1))
        self.bias = nn.Parameter(torch.zeros(max_in_way + 1))


    def forward(self,
                input_tensors: torch.Tensor,
                ) -> torch.Tensor:
        # input_tensors:
        #   0: [n_batch, n_atoms, n_channel]
        #   1: [n_batch, n_atoms, n_channel, n_dim]
        #   2: [n_batch, n_atoms, n_channel, n_dim, n_dim]
        #   .....
        output_tensors = {}
        for way in input_tensors:
            if way == 0:
                output_tensor = self.activate_fn(self.weights[way] * input_tensors[way] + self.bias[way])
            else:
                norm = torch.linalg.norm(input_tensors[way], dim=tuple(range(3, 3 + way)), keepdim=True)
                factor = self.activate_fn(self.weights[way] * norm + self.bias[way])
                output_tensor = factor * input_tensors[way]
            output_tensors[way] = output_tensor
            ###################################################################
            # print("Non")
            # print(way)
            # if way == 0:
            #     print(output_tensor[0] - output_tensor[1])
            # if way == 1:
            #     R1 = output_tensor[0, :, 0]
            #     R2 = output_tensor[1, :, 0]
            #     print(R1 @ R.T - R2)
            # if way == 2:
            #     T1 = output_tensor[0, :, 0]
            #     T2 = output_tensor[1, :, 0]
            #     print(T2 - torch.matmul(torch.matmul(R, T1), R.T))
            ##################################################################
        return output_tensors


class SOnEquivalentLayer(nn.Module):
    def __init__(self,
                 activate_fn : Callable,
                 radius_fn   : RadiusFunction,
                 max_r_way   : int,
                 max_out_way : int,
                 input_dim   : int,
                 output_dim  : int,
                 ) -> None:
        super().__init__()
        self.tensor_aggregate = TensorAggregateLayer(radius_fn=radius_fn,
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
                coordinate    : torch.Tensor,
                neighbor      : torch.Tensor,
                mask          : torch.Tensor,
                cell          : Optional[torch.Tensor]=None,
                offset        : Optional[torch.Tensor]=None,
                ) -> Dict[int, torch.Tensor]:
        # input_tensors:
        #   0: [n_batch, n_atoms, n_channel]
        #   1: [n_batch, n_atoms, n_channel, n_dim]
        #   2: [n_batch, n_atoms, n_channel, n_dim, n_dim]
        #   .....
        output_tensors = self.tensor_aggregate(input_tensors=input_tensors, 
                                               coordinate=coordinate,
                                               neighbor=neighbor,
                                               mask=mask,
                                               cell=cell,
                                               offset=offset)
        output_tensors = self.self_interact(output_tensors)
        output_tensors = self.non_linear(output_tensors)
        return output_tensors
