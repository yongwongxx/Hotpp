import site
from typing import Callable, List, Dict, Optional, Literal
from numpy import squeeze
import torch
from torch import nn

from hotpp.layer.radial import ChebyshevPoly
from .base import AtomicModule
from ..layer import EmbeddingLayer, RadialLayer, ReadoutLayer
from ..layer.equivalent import NonLinearLayer, GraphConvLayer, SelfInteractionLayer, TensorProductLayer
from ..utils import find_distances, _scatter_add, res_add, find_spin, find_moment, find_spin_moment, expand_to
from .miao import MiaoBlock


# class SpinConvLayer(nn.Module):
#     def __init__(self, 
#                  radial_fn      : RadialLayer,
#                  spin_radial_fn : RadialLayer,
#                  input_dim      : int,
#                  output_dim     : int,
#                  max_in_way     : int=2,
#                  max_r_way      : int=2,
#                  max_m_way      : int=2,
#                  max_out_way    : int=2,
#                  conv_mode      : Literal['node_j', 'node_edge']='node_j',
#                  ) -> None:
#         super().__init__()
#         self.radial_fn = radial_fn
#         self.spin_radial_fn = spin_radial_fn
#         self.rbf_mixing_list = nn.ModuleList([
#             nn.Linear(radial_fn.n_channel, output_dim, bias=False)
#             for r_way in range(max_r_way + 1)
#         ])
#         self.spin_rbf_mixing_list = nn.ModuleList([
#             nn.Linear(spin_radial_fn.n_channel, output_dim, bias=False)
#             for m_way in range(max_m_way + 1)
#         ])
#         self.U = SelfInteractionLayer(input_dim=input_dim,
#                                       max_way=max_in_way,
#                                       output_dim=output_dim)
#         # self.spin_spin_product = TensorProductLayer(max_x_way=max_m_way,
#         #                                             max_y_way=max_m_way,
#         #                                             max_z_way=max_m_way)
#         # self.node_spin_product = TensorProductLayer(max_x_way=max_in_way,
#         #                                             max_y_way=max_m_way,
#         #                                             max_z_way=max_out_way)
#         # self.nodespin_r_product = TensorProductLayer(max_x_way=max_out_way,
#         #                                          max_y_way=max_r_way,
#         #                                          max_z_way=max_out_way)
#         self.max_in_way = max_in_way

#     def forward(self,
#                 node_info  : Dict[int, torch.tensor],
#                 edge_info  : Dict[int, torch.tensor],
#                 batch_data : Dict[str, torch.tensor]):
#         idx_i = batch_data['idx_i']
#         idx_j = batch_data['idx_j']
#         _, dij, _ = find_distances(batch_data)
#         rbf_ij = self.radial_fn(dij)
#         mi, _ = find_spin(batch_data)
#         spin_rbf_i = self.spin_radial_fn(mi)
#         node = {}
#         for in_way in range(self.max_in_way + 1):
#             node[in_way] = node_info[in_way][idx_j]
#         node = self.U(node)

#         spin_i, spin_j = {}, {}
#         for m_way, spin_rbf_mixing in enumerate(self.spin_rbf_mixing_list):
#             spin_m = find_spin_moment(batch_data, m_way).unsqueeze(1) * \
#                 expand_to(spin_rbf_mixing(spin_rbf_i), n_dim=m_way + 2)
#             spin_i[m_way] = spin_m[idx_i]
#             spin_j[m_way] = spin_m[idx_j]
#         spin = self.spin_spin_product(spin_i, spin_j)
#         nodespin = self.node_spin_product(node, spin)
#         r = {}
#         for r_way, rbf_mixing in enumerate(self.rbf_mixing_list):
#             r[r_way] = find_moment(batch_data, r_way).unsqueeze(1) * \
#                 expand_to(rbf_mixing(rbf_ij), n_dim=r_way + 2)
#         return self.nodespin_r_product(nodespin, r)
    
    
class SpinConvLayer(nn.Module):
    def __init__(self, 
                 radial_fn      : RadialLayer,
                 spin_radial_fn : RadialLayer,
                 input_dim      : int,
                 output_dim     : int,
                 max_in_way     : int=2,
                 max_r_way      : int=2,
                 max_m_way      : int=2,
                 max_out_way    : int=2,
                 ) -> None:
        super().__init__()
        self.radial_fn = radial_fn
        self.n_cheb = 3
        self.sij_basis_fn = ChebyshevPoly(r_max=1., r_min=-1., n_max=self.n_cheb)
        self.spin_radial_fn = spin_radial_fn
        self.rbf_mixing_list = nn.ModuleList([
            nn.Linear(radial_fn.n_channel, output_dim * self.n_cheb)
            for r_way in range(max_r_way + 1)
        ])
        self.spin_rbf_mixing_list = nn.ModuleList([
            nn.Linear(spin_radial_fn.n_channel, output_dim * self.n_cheb)
            for m_way in range(max_r_way + 1)
        ])
        self.U = SelfInteractionLayer(input_dim=input_dim,
                                      max_way=max_in_way,
                                      output_dim=output_dim)
        self.V = SelfInteractionLayer(input_dim=self.n_cheb * output_dim,
                                      max_way=max_r_way,
                                      output_dim=output_dim)

        # 如果不考虑自旋轨道耦合，输出的维度应该只有0维，
        # 此时结果中只会出现 sisj 的多项式，如max_m_way=1则包含f0(mi)f0(mj)与f1(mi)f1(mj)sisj
        # 如max_m_way=2则包含f0(mi)f0(mj)与f1(mi)f1(mj)sisj与f2(mi)f2(mj)(sisj)2 （两个并矢的双点积）
        # 既然如此可以将spin_spin_product替换成径向函数与sisj函数的直积

        self.node_sr_product = TensorProductLayer(max_x_way=max_in_way,
                                                  max_y_way=max_r_way,
                                                  max_z_way=max_out_way)
        self.max_in_way = max_in_way

    def forward(self,
                node_info  : Dict[int, torch.tensor],
                edge_info  : Dict[int, torch.tensor],
                batch_data : Dict[str, torch.tensor]):
        idx_i = batch_data['idx_i']
        idx_j = batch_data['idx_j']
        n_edge = len(idx_i)
        _, dij, _ = find_distances(batch_data)
        rbf_ij = self.radial_fn(dij)
        mi, si, sij = find_spin(batch_data)
        spin_rbf_i = self.spin_radial_fn(mi)
        node = {}
        for in_way in range(self.max_in_way + 1):
            node[in_way] = node_info[in_way][idx_j]
        node = self.U(node)

        spin_basis = self.sij_basis_fn(sij)    # [n_edge, n_cheb]
        r = {}
        for way, (rbf_mixing, spin_rbf_mixing) in enumerate(zip(self.rbf_mixing_list, self.spin_rbf_mixing_list)):
            sb = spin_rbf_mixing(spin_rbf_i)      # [n_node, n_channel]
            hehe = rbf_mixing(rbf_ij) * sb[idx_i] * sb[idx_j]     # f(rij)g(mi)g(mj)  [n_edge, n_channel]
            xixi = hehe.view(n_edge, self.n_cheb, -1) * spin_basis.unsqueeze(-1)
            r[way] = find_moment(batch_data, way).unsqueeze(1) * expand_to(xixi.view(n_edge, -1), n_dim=way + 2)
        r = self.V(r)
        return self.node_sr_product(node, r)


class SpinMiaoBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 spin_radial_fn : RadialLayer,
                 max_r_way      : int,
                 max_m_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 norm_factor    : float=1.0,
                 activate_fn    : str='silu',
                 ) -> None:
        super().__init__()
        self.graph_conv = SpinConvLayer(radial_fn=radial_fn,
                                        spin_radial_fn=spin_radial_fn,
                                        input_dim=input_dim,
                                        output_dim=output_dim,
                                        max_in_way=max_in_way,
                                        max_out_way=max_out_way,
                                        max_r_way=max_r_way,
                                        max_m_way=max_m_way,
                                        )
        self.self_interact = SelfInteractionLayer(input_dim=input_dim,
                                                  max_way=max_out_way,
                                                  output_dim=output_dim)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_way=max_out_way,
                                         input_dim=output_dim)
        self.register_buffer("norm_factor", torch.tensor(norm_factor))

    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        if len(batch_data['idx_i']) > 0:
            message = self.graph_conv(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
            res_info = {}
            idx_i = batch_data["idx_i"]
            n_atoms = batch_data['atomic_number'].shape[0]
            for way in message.keys():
                res_info[way] = _scatter_add(message[way], idx_i, dim_size=n_atoms) / self.norm_factor
            res_info = self.non_linear(self.self_interact(res_info))
            return res_add(node_info, res_info), edge_info
        else:
            return node_info, edge_info

class SpinMiaoNet(AtomicModule):

    def __init__(self,
                 embedding_layer : EmbeddingLayer,
                 radial_fn       : RadialLayer,
                 spin_radial_fn  : RadialLayer,
                 n_layers        : int,
                 n_spin_layers   : int,
                 max_r_way       : List[int],
                 max_m_way       : List[int],
                 max_out_way     : List[int],
                 output_dim      : List[int],
                 activate_fn     : str="silu",
                 target_way      : Dict[str, int]={"site_energy": 0},
                 mean            : float=0.,
                 std             : float=1.,
                 norm_factor     : float=1.,
                 bilinear        : bool=False,
                 conv_mode       : Literal['node_j', 'node_edge']='node_j',
                 update_edge     : bool=False,
                 ):
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean).float())
        self.register_buffer("std", torch.tensor(std).float())
        self.embedding_layer = embedding_layer
        self.radial_fn = radial_fn

        max_in_way = [0] + max_out_way[1:]
        hidden_nodes = [embedding_layer.n_channel] + output_dim
        self.en_equivalent_blocks = nn.ModuleList([
            MiaoBlock(activate_fn=activate_fn,
                      radial_fn=radial_fn.replicate(),
                      # Use factory method, so the radial_fn in each layer are different
                      max_r_way=max_r_way[i],
                      max_in_way=max_in_way[i],
                      max_out_way=max_out_way[i],
                      input_dim=hidden_nodes[i],
                      output_dim=hidden_nodes[i + 1],
                      norm_factor=norm_factor,
                      conv_mode=conv_mode,
                      update_edge=update_edge,
                      ) for i in range(n_layers)])

        self.en_equivalent_spin_blocks = nn.ModuleList([
            SpinMiaoBlock(activate_fn=activate_fn,
                      radial_fn=radial_fn.replicate(),
                      spin_radial_fn=spin_radial_fn.replicate(),
                      # Use factory method, so the radial_fn in each layer are different
                      max_r_way=max_r_way[i + n_layers],
                      max_m_way=max_m_way[i],
                      max_in_way=max_in_way[i + n_layers],
                      max_out_way=max_out_way[i + n_layers],
                      input_dim=hidden_nodes[i + n_layers],
                      output_dim=hidden_nodes[i + n_layers + 1],
                      norm_factor=norm_factor,
                      ) for i in range(n_spin_layers)])
        self.readout_layer = ReadoutLayer(n_dim=hidden_nodes[-1],
                                          target_way=target_way,
                                          activate_fn=activate_fn,
                                          bilinear=bilinear,
                                          e_dim=embedding_layer.n_channel)
        self.spin_init = nn.Sequential(
            spin_radial_fn,
            nn.Linear(spin_radial_fn.n_channel, hidden_nodes[n_layers], bias=False)
            )

    def calculate(self,
                  batch_data : Dict[str, torch.Tensor],
                  ) -> Dict[str, torch.Tensor]:
        node_info, edge_info = self.get_init_info(batch_data)
        for en_equivalent in self.en_equivalent_blocks:
            node_info, edge_info = en_equivalent(node_info, edge_info, batch_data)

        mi, _, _ = find_spin(batch_data)
        node_info[0] = node_info[0] + self.spin_init(mi)
        for en_equivalent in self.en_equivalent_spin_blocks:
            node_info, edge_info = en_equivalent(node_info, edge_info, batch_data)

        output_tensors = self.readout_layer(node_info, None)
        if 'site_energy' in output_tensors:
            output_tensors['site_energy'] = output_tensors['site_energy'] * self.std + self.mean
        if 'direct_forces' in output_tensors:
            output_tensors['direct_forces'] = output_tensors['direct_forces'] * self.std
        return output_tensors

    def get_init_info(self,
                      batch_data : Dict[str, torch.Tensor],
                      ):
        emb = self.embedding_layer(batch_data=batch_data)
        node_info = {0: emb}
        _, dij, _ = find_distances(batch_data)
        rbf = self.radial_fn(dij)
        edge_info = {0: rbf}
        return node_info, edge_info
