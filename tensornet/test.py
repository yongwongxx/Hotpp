from scipy.io import loadmat
import torch
from torch import nn
from utils import setup_seed, way_combination, expand_to, multi_outer_product, find_distances


setup_seed(0)


def tensor_aggregate_layer(
        input_tensors :     dict, 
        coordinate    :     torch.Tensor, 
        max_out_way   :     int=-1, 
        max_r_way     :     int=2):
    # input_tensors:
    #   0: [n_batch, n_atoms, n_channel]
    #   1: [n_batch, n_atoms, n_channel, n_dim]
    #   2: [n_batch, n_atoms, n_channel, n_dim, n_dim]
    #   .....
    # coordinate: [n_batch, n_atoms, n_dim]
    if max_out_way == -1:
        max_out_way = len(input_tensors) - 1
    output_tensors = {way: None for way in range(max_out_way + 1)}

    for out_way, in_way, r_way in way_combination(max_out_way + 1, len(input_tensors), max_r_way + 1):
        coupling_way = (in_way + r_way - out_way) // 2
        rij = find_distances(coordinate)
        filter_tensor = multi_outer_product(rij, r_way)
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
        # Debug: SO(n) symmetry of each tensor
        # if out_way == 0:
        #     print(output_tensor[0] - output_tensor[1])
        # if out_way == 1:
        #     R1 = output_tensor[0, :, 0]
        #     R2 = output_tensor[1, :, 0]
        #     print(R1 @ tR.T - R2)
        # if out_way == 2:
        #     T1 = output_tensor[0, :, 0]
        #     T2 = output_tensor[1, :, 0]
        #     print(T2 - torch.matmul(torch.matmul(tR, T1), tR.T))
        ###################################################################
    return output_tensors


def self_interaction_layer(
        input_tensors,
        output_dim):
    # input_tensors:
    #   0: [n_batch, n_atoms, n_channel]
    #   1: [n_batch, n_atoms, n_channel, n_dim]
    #   2: [n_batch, n_atoms, n_channel, n_dim, n_dim]
    #   .....
    output_tensors = {}
    for way in input_tensors:
        # swap channel axis and the last dim axis
        input_dim = input_tensors[way].shape[2]
        input_tensor = torch.transpose(input_tensors[way], 2, -1)
        i_to_o = nn.Linear(input_dim, output_dim, bias=False)
        output_tensor = i_to_o(input_tensor)
        output_tensors[way] = torch.transpose(output_tensor, 2, -1)
        ###################################################################
        # Debug: SO(n) symmetry of each tensor
        # if way == 0:
        #     print(output_tensors[way][0] - output_tensors[way][1])
        # if way == 1:
        #     R1 = output_tensors[way][0, :, 0]
        #     R2 = output_tensors[way][1, :, 0]
        #     print(R1 @ tR.T - R2)
        # if way == 2:
        #     T1 = output_tensors[way][0, :, 0]
        #     T2 = output_tensors[way][1, :, 0]
        #     print(T2 - torch.matmul(torch.matmul(tR, T1), tR.T))
        ###################################################################
    return output_tensors


def nonlinear_layer(
        input_tensors: torch.Tensor,
        activate_fn: nn.functional):
    # input_tensors:
    #   0: [n_batch, n_atoms, n_channel]
    #   1: [n_batch, n_atoms, n_channel, n_dim]
    #   2: [n_batch, n_atoms, n_channel, n_dim, n_dim]
    #   .....
    output_tensors = {}
    for way in input_tensors:
        if way == 0:
            output_tensor = activate_fn(input_tensors[way])
        else:
            norm = torch.linalg.norm(input_tensors[way], dim=tuple(range(3, 3 + way)), keepdim=True)
            factor = activate_fn(norm)
            output_tensor = factor * input_tensors[way]
        output_tensors[way] = output_tensor
    return output_tensors
    

datapath = '../dataset/qm7.mat'
raw_data = loadmat(datapath)
dim = 3
max_in_way = 2        # 0: scalar  1: vector  2: matrix  ...

symbol = raw_data['Z']           # nb, na
coordinate = torch.tensor(raw_data['R'][:2, :4])   # nb, na, dim

from scipy.stats import special_ortho_group
R = special_ortho_group.rvs(dim=dim)
tR = torch.tensor(R, dtype=torch.float32)
coordinate[1] = coordinate[0] @ R.T

input_tensors = {}
for way in range(max_in_way + 1):
    t = multi_outer_product(coordinate, way)
    input_tensors[way] = torch.unsqueeze(t, 2)

output_tensors = tensor_aggregate_layer(input_tensors , coordinate, max_out_way=2)
output_tensors = self_interaction_layer(output_tensors, output_dim=5)
output_tensors = nonlinear_layer(output_tensors, torch.sigmoid)

output_tensors = tensor_aggregate_layer(output_tensors, coordinate, max_out_way=2)
output_tensors = self_interaction_layer(output_tensors, output_dim=5)
output_tensors = nonlinear_layer(output_tensors, torch.sigmoid)

output_tensors = tensor_aggregate_layer(output_tensors, coordinate, max_out_way=0)
output_tensors = self_interaction_layer(output_tensors, output_dim=1)
output_tensors = nonlinear_layer(output_tensors, torch.sigmoid)


print(output_tensors[0][0] - output_tensors[0][1])
R1 = output_tensors[1][0, :, 0]
R2 = output_tensors[1][1, :, 0]
print(R1 @ tR.T - R2)

T1 = output_tensors[2][0, :, 0]
T2 = output_tensors[2][1, :, 0]
print(T2 - torch.matmul(torch.matmul(tR, T1), tR.T))