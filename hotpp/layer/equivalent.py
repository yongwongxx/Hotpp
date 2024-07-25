# TODO 
# resnet
# Now segment_coo have some bug in getting second-order derivative
# https://github.com/rusty1s/pytorch_scatter/issues/299
# from torch_scatter import segment_coo
# Different channel for different order?
# parameterdict cannot be scripted
# https://github.com/pytorch/pytorch/issues/128496

import torch
from torch import nn
from typing import Dict, Callable, Literal, Union, Optional
from .base import RadialLayer, CutoffLayer
from .activate import TensorActivateDict
from ..utils import find_distances, find_moment, _scatter_add, _aggregate, expand_to, TensorAggregateOP, _aggregate_new


# input_tensors be like:
#   0: [n_atoms, n_channel]
#   1: [n_atoms, n_channel, n_dim]
#   2: [n_atoms, n_channel, n_dim, n_dim]
#   .....
# coordinate: [n_atoms, n_dim]

__all__ = ["TensorLinear",
           "TensorBiLinear",
           "TensorAggregateLayer",
           "SelfInteractionLayer",
           "NonLinearLayer",
           "MultiBodyLayer",
           ]


class TensorLinear(nn.Module):
    """
    Linear layer for tensors with shape [n_batch, n_channel, n_dim, n_dim, ...]
    """
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
        input_tensor = torch.transpose(input_tensor, 1, -1)
        output_tensor = self.linear(input_tensor)
        output_tensor = torch.transpose(output_tensor, 1, -1)
        return output_tensor


class TensorBiLinear(nn.Module):
    """
    Linear layer for tensors with shape [n_batch, n_channel, n_dim, n_dim, ...]
    """
    def __init__(self,
                 input_dim   : int,
                 emb_dim     : int,
                 output_dim  : int,
                 bias        : bool=False,
                 ) -> None:
        super().__init__()
        self.linear = nn.Bilinear(input_dim, emb_dim, output_dim, bias=bias)

    def forward(self,
                input_tensor: torch.Tensor,   # [n_batch, n_channel, n_dim, n_dim, ...]
                emb:          torch.Tensor,   # [n_batch, n_emb_channel]
                ):
        way = len(input_tensor.shape) - 2
        input_tensor = torch.transpose(input_tensor, 1, -1)
        shape = list(input_tensor.shape)
        shape[-1] = -1
        emb = expand_to(emb, way + 2, dim=1).expand(shape)
        output_tensor = self.linear(input_tensor, emb)
        output_tensor = torch.transpose(output_tensor, 1, -1)
        return output_tensor


class SimpleTensorAggregateLayer(nn.Module):
    """
    In this type of layer, the rbf mixing only different if r_way different
    """
    def __init__(self,
                 radial_fn      : RadialLayer,
                 n_channel      : int,
                 max_in_way     : int=2,
                 max_out_way    : int=2,
                 max_r_way      : int=2,
                 norm_factor    : float=1.,
                 ) -> None:
        super().__init__()
        # get all possible "i, r, o" combinations
        self.rbf_mixing_dict = nn.ModuleDict()
        self.max_r_way = max_r_way
        self.inout_combinations = {r_way: [] for r_way in range(max_r_way + 1)}
        for r_way in range(max_r_way + 1):
            self.rbf_mixing_dict[str(r_way)] = nn.Linear(radial_fn.n_channel, n_channel, bias=True)
            for in_way in range(max_in_way + 1):
                for z_way in range(min(in_way, r_way) + 1):
                    out_way = in_way + r_way - 2 * z_way
                    if out_way <= max_out_way:
                        self.inout_combinations[r_way].append((in_way, out_way))
        self.radial_fn = radial_fn
        self.register_buffer("norm_factor", torch.tensor(norm_factor).float())

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                batch_data    : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        # These 3 rows are required by torch script
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        idx_i = batch_data['idx_i']
        idx_j = batch_data['idx_j']

        n_atoms = batch_data['atomic_number'].shape[0]
        _, dij, uij = find_distances(batch_data)
        rbf_ij = self.radial_fn(dij)    # [n_edge, n_rbf]

        for r_way, rbf_mixing in self.rbf_mixing_dict.items():
            r_way = int(r_way)
            fn = rbf_mixing(rbf_ij) # [n_edge, n_channel]
            # TODO: WHY!!!!!!!!!! CAO!
            # fn = fn * input_tensor_dict[0]
            moment_tensor = find_moment(batch_data, r_way)  # [n_edge, n_dim, ...]
            for in_way, out_way in self.inout_combinations[r_way]:
                input_tensor = input_tensors[in_way][idx_j]      # [n_edge, n_channel, n_dim, n_dim, ...]
                output_tensor = _aggregate(moment_tensor, fn, input_tensor, in_way, r_way, out_way)
                output_tensor = _scatter_add(output_tensor, idx_i, dim_size=n_atoms) / self.norm_factor

                if out_way not in output_tensors:
                    output_tensors[out_way] = output_tensor
                else:
                    output_tensors[out_way] += output_tensor
        return output_tensors


class TensorAggregateLayer(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 n_channel      : int,
                 max_in_way     : int=2,
                 max_out_way    : int=2,
                 max_r_way      : int=2,
                 norm_factor    : float=1.,
                 ) -> None:
        super().__init__()
        # get all possible "i, r, o" combinations
        self.all_combinations = {}
        self.rbf_mixing_dict = nn.ModuleDict()
        for in_way in range(max_in_way + 1):
            for r_way in range(max_r_way + 1):
                for z_way in range(min(in_way, r_way) + 1):
                    out_way = in_way + r_way - 2 * z_way
                    if out_way <= max_out_way:
                        comb = (in_way, r_way, out_way)
                        self.all_combinations[str(comb)] = comb
                        self.rbf_mixing_dict[str(comb)] = nn.Linear(radial_fn.n_channel, n_channel, bias=True)

        self.radial_fn = radial_fn
        self.register_buffer("norm_factor", torch.tensor(norm_factor).float())

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                batch_data    : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        # These 3 rows are required by torch script
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        idx_i = batch_data['idx_i']
        idx_j = batch_data['idx_j']

        n_atoms = batch_data['atomic_number'].shape[0]
        _, dij, uij = find_distances(batch_data)
        rbf_ij = self.radial_fn(dij)    # [n_edge, n_rbf]

        for comb, rbf_mixing in self.rbf_mixing_dict.items():
            in_way, r_way, out_way = self.all_combinations[comb]
            fn = rbf_mixing(rbf_ij) # [n_edge, n_channel]
            # TODO: WHY!!!!!!!!!! CAO!
            # fn = fn * input_tensor_dict[0]
            moment_tensor = find_moment(batch_data, r_way)  # [n_edge, n_dim, ...]
            input_tensor = input_tensors[in_way][idx_j]      # [n_edge, n_channel, n_dim, n_dim, ...]
            output_tensor = _aggregate(moment_tensor, fn, input_tensor, in_way, r_way, out_way)
            output_tensor = _scatter_add(output_tensor, idx_i, dim_size=n_atoms) / self.norm_factor
            if out_way not in output_tensors:
                output_tensors[out_way] = output_tensor
            else:
                output_tensors[out_way] += output_tensor

        return output_tensors


class SelfInteractionLayer(nn.Module):
    def __init__(self,
                 input_dim  : int,
                 max_way : int,
                 output_dim : int=10,
                 ) -> None:
        super().__init__()
        # only the way 0 can have bias
        self.linear_list = nn.ModuleList([
            nn.Linear(input_dim, output_dim, bias=(way==0)) for way in range(max_way + 1)])

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for way, linear in enumerate(self.linear_list):
            if way in input_tensors:
                # swap channel axis and the last dim axis
                input_tensor = torch.transpose(input_tensors[way], 1, -1)
                output_tensor = linear(input_tensor)
                output_tensors[way] = torch.transpose(output_tensor, 1, -1)
        return output_tensors


# TODO: cat different way together and use Linear layer to got factor of every channel
class NonLinearLayer(nn.Module):
    def __init__(self,
                 max_way  : int,
                 input_dim   : int,
                 activate_fn : str='jilu',
                 ) -> None:
        super().__init__()
        self.activate_list = nn.ModuleList([TensorActivateDict[activate_fn](input_dim) for _ in range(max_way + 1)])

    def forward(self,
                input_tensors: Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for way, activate in enumerate(self.activate_list):
            if way in input_tensors:
                output_tensors[way] = activate(input_tensors[way])
        return output_tensors

# TODO: test two methods:
# for different path to the same l (l1, l2 -> l and l3, l4 -> l), use coefficient or a large linear layer
# number parameters: n_path + n_channel * n_channel vs. npath * n_channel * n_channel
class TensorProductLayer(nn.Module):

    def __init__(self,
                 max_x_way      : int=2,
                 max_y_way      : int=2,
                 max_z_way      : int=2,
                 ) -> None:
        super().__init__()
        self.combinations = []
        #self.coefficient = nn.ParameterDict()
        for x_way in range(max_x_way + 1):
            for y_way in range(max_y_way + 1):
                for z_way in range(abs(y_way - x_way), min(max_z_way, x_way + y_way) + 1, 2):
                    self.combinations.append((x_way, y_way, z_way))
                    #self.coefficient[str((x_way, y_way, z_way))] = nn.Parameter(torch.tensor(1.))

    def forward(self,
                x : Dict[int, torch.Tensor],
                y : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for x_way, y_way, z_way in self.combinations:
            if x_way not in x or y_way not in y:
                continue
            #coef: torch.Tensor = self.coefficient[str((x_way, y_way, z_way))]
            #output_tensor = coef * _aggregate_new(x[x_way], y[y_way], x_way, y_way, z_way)
            output_tensor = _aggregate_new(x[x_way], y[y_way], x_way, y_way, z_way)
            if z_way not in output_tensors:
                output_tensors[z_way] = output_tensor
            else:
                output_tensors[z_way] += output_tensor
        return output_tensors


# TODO
# Test allowing higher order such as (2, 2) -> 4, (4, 2) -> 2 ?
class MultiBodyLayer(nn.Module):
    """
    Node operation:
    mix h_i^l for different l
    """
    def __init__(self,
                 input_dim      : int,
                 output_dim     : int,
                 max_n_body     : int=3,
                 max_way     : int=2,
                 ) -> None:
        super().__init__()
        self.max_n_body = max_n_body
        self.max_way = max_way
        n_body_tensors = [[1] *  (max_way + 1)]
        for n in range(max_n_body - 1):
            self.combinations = []
            n_body_tensors.append([0] *  (max_way + 1))
            for way1 in range(max_way + 1):
                for way2 in range(way1, max_way + 1):
                    for way3 in range(abs(way2 - way1), min(max_way, way1 + way2) + 1, 2):
                        n_body_tensors[n + 1][way3] += n_body_tensors[n][way1]
                        self.combinations.append((way1, way2, way3))

        self.linear_list = nn.ModuleList([
            TensorLinear(input_dim * sum([n_body_tensors[n][way] for n in range(max_n_body)]), 
                         output_dim, 
                         bias=(way==0)) 
            for way in range(max_way + 1)])

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = {}
        n_body_tensors = {0: {way: [input_tensors[way]] for way in input_tensors}}
        for n in range(self.max_n_body - 1):
            n_body_tensors[n + 1] = {way: [] for way in range(self.max_way + 1)}
            for way1, way2, way3 in self.combinations:
                for tensor in n_body_tensors[n][way1]:
                    n_body_tensors[n + 1][way3].append(
                        _aggregate_new(tensor, input_tensors[way2], way1, way2, way3)
                        # TensorAggregateOP.oplist[(way1, way2, way3)](tensor, 
                        #                                              input_tensors[way2],
                        #                                              way1, 
                        #                                              way2, 
                        #                                              way3)
                        )
        for way, linear in enumerate(self.linear_list):
            tensor = torch.cat([t for n in range(self.max_n_body) for t in n_body_tensors[n][way]], dim=1)  # nb, nc*n, nd, nd, ...
            output_tensors[way] = linear(tensor)
        return output_tensors


class GraphConvLayer(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 input_dim      : int,
                 output_dim     : int,
                 max_in_way     : int=2,
                 max_r_way      : int=2,
                 max_out_way    : int=2,
                 conv_mode      : Literal['node_j', 'node_edge']='node_j',
                 ) -> None:
        """Graph convolution layer

        Args:
            radial_fn (RadialLayer): rdf
            input_dim (int): number of input channel
            output_dim (int): number of output channel
            max_in_way (int, optional): max order of input tensors. Defaults to 2.
            max_r_way (int, optional): max order of filter tensors. Defaults to 2.
            max_out_way (int, optional): max order of output tensors. Defaults to 2.
            mode ('node_j' or 'node_edge', optional):
                'node_j' use node_info['idx_j'] only.
                'node_edge' use node_info['idx_i'] | node_info['idx_j'] | edge_info.
                Defaults to 'node_j'.
        """
        super().__init__()
        self.radial_fn = radial_fn
        self.rbf_mixing_list = nn.ModuleList([
            nn.Linear(radial_fn.n_channel, output_dim, bias=True)
            for r_way in range(max_r_way + 1)
        ])
        if conv_mode == 'node_j':
            self.U = SelfInteractionLayer(input_dim=input_dim,
                                          max_way=max_in_way,
                                          output_dim=output_dim)
        elif conv_mode == 'node_edge':
            self.U = SelfInteractionLayer(input_dim=input_dim * 3,
                                          max_way=max_in_way,
                                          output_dim=output_dim)
        self.tensor_product = TensorProductLayer(max_x_way=max_in_way,
                                                 max_y_way=max_r_way,
                                                 max_z_way=max_out_way)
        self.max_in_way = max_in_way
        self.max_r_way = max_r_way
        self.conv_mode = conv_mode

    def forward(self,
                node_info  : Dict[int, torch.Tensor],
                edge_info  : Dict[int, torch.Tensor],
                batch_data : Dict[str, torch.Tensor]):
        idx_i = batch_data['idx_i']
        idx_j = batch_data['idx_j']
        _, dij, _ = find_distances(batch_data)
        rbf_ij = self.radial_fn(dij)
        x = torch.jit.annotate(Dict[int, torch.Tensor], {})
        y = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for in_way in range(self.max_in_way + 1):
            if self.conv_mode == 'node_j':
                x[in_way] = node_info[in_way][idx_j]
            elif self.conv_mode == 'node_edge':
                x[in_way] = torch.cat([node_info[in_way][idx_i],
                                       node_info[in_way][idx_j],
                                       edge_info[in_way]], dim=1)
        x = self.U(x)

        for r_way, rbf_mixing in enumerate(self.rbf_mixing_list):
            fn = rbf_mixing(rbf_ij)
            y[r_way] = find_moment(batch_data, r_way).unsqueeze(1) * expand_to(fn, n_dim=r_way + 2)

        return self.tensor_product(x, y)


class GraphNorm(nn.Module):
    """
    GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training
    加了怎么还变垃圾了？
    """
    def __init__(
        self,
        max_way: int,
        n_channel: int,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.empty(max_way + 1, n_channel))
        self.beta = nn.Parameter(torch.empty(n_channel))   # only way=0 can add bias
        self.gamma = nn.Parameter(torch.empty(max_way + 1, n_channel))

        self.reset_parameters()

    def reset_parameters(self):
        self.alpha.data.fill_(1.)
        self.gamma.data.fill_(1.)
        self.beta.data.fill_(0.)

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                batch         : torch.Tensor,
                degree        : torch.Tensor,
                ) -> Dict[int, torch.Tensor]:

        output_tensors = {}
        for way in input_tensors:
            output_tensor = input_tensors[way]

            mean = _scatter_add(output_tensor, batch) / expand_to(degree, way + 2)
            output_tensor = output_tensor - mean[batch] * expand_to(self.alpha[way][None, :], way + 2)
            var = _scatter_add(output_tensor ** 2, batch) / expand_to(degree, way + 2)
            output_tensor = output_tensor / (var[batch] + self.eps).sqrt()

            output_tensor = output_tensor * expand_to(self.gamma[way][None, :], way + 2)
            if way == 0:
                output_tensor = output_tensor + self.beta[None, :]
            output_tensors[way] = output_tensor
        return output_tensors
