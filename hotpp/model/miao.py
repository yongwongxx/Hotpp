from typing import Callable, List, Dict, Optional, Literal, Tuple
import torch
from torch import nn
from .base import AtomicModule
from ..layer import EmbeddingLayer, RadialLayer, ReadoutLayer
from ..layer.equivalent import NonLinearLayer, GraphConvLayer, SelfInteractionLayer
from ..utils import find_distances, _scatter_add, res_add, TensorAggregateOP


class UpdateNodeBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 norm_factor    : float=1.0,
                 activate_fn    : str='silu',
                 conv_mode      : Literal['node_j', 'node_edge']='node_j',
                 ) -> None:
        super().__init__()
        self.graph_conv = GraphConvLayer(radial_fn=radial_fn,
                                         input_dim=input_dim,
                                         output_dim=output_dim,
                                         max_in_way=max_in_way,
                                         max_out_way=max_out_way,
                                         max_r_way=max_r_way,
                                         conv_mode=conv_mode,
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
        message = self.graph_conv(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        res_info = torch.jit.annotate(Dict[int, torch.Tensor], {})
        idx_i = batch_data["idx_i"]
        n_atoms = batch_data['atomic_number'].shape[0]
        for way in message.keys():
            res_info[way] = _scatter_add(message[way], idx_i, dim_size=n_atoms) / self.norm_factor
        res_info = self.non_linear(self.self_interact(res_info))
        return res_add(node_info, res_info)


class UpdateEdgeBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 activate_fn    : str='silu',
                 conv_mode      : Literal['node_j', 'node_edge']='node_j',
                 ) -> None:
        super().__init__()
        self.graph_conv = GraphConvLayer(radial_fn=radial_fn,
                                         input_dim=input_dim,
                                         output_dim=output_dim,
                                         max_in_way=max_in_way,
                                         max_out_way=max_out_way,
                                         max_r_way=max_r_way,
                                         conv_mode=conv_mode)
        self.self_interact = SelfInteractionLayer(input_dim=input_dim,
                                                  max_way=max_out_way,
                                                  output_dim=output_dim)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_way=max_out_way,
                                         input_dim=output_dim)

    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        message = self.graph_conv(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        res_info = self.non_linear(self.self_interact(message))
        return res_add(edge_info, res_info)

class MiaoBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 norm_factor    : float=1.0,
                 activate_fn    : str='silu',
                 conv_mode      : Literal['node_j', 'node_edge']='node_j',
                 update_edge    : bool=False,
                 ) -> None:
        super().__init__()
        self.node_block = UpdateNodeBlock(radial_fn=radial_fn, 
                                          max_r_way=max_r_way, 
                                          max_in_way=max_in_way,
                                          max_out_way=max_out_way,
                                          input_dim=input_dim, 
                                          output_dim=output_dim, 
                                          norm_factor=norm_factor, 
                                          activate_fn=activate_fn,
                                          conv_mode=conv_mode,
                                          )
        if update_edge:
            self.edge_block = UpdateEdgeBlock(radial_fn=radial_fn, 
                                              max_r_way=max_r_way, 
                                              max_in_way=max_in_way,
                                              max_out_way=max_out_way,
                                              input_dim=input_dim,
                                              output_dim=output_dim,
                                              activate_fn=activate_fn,
                                              conv_mode=conv_mode,
                                              )
        else:
            self.edge_block = None

    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        node_info = self.node_block(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        if self.edge_block is not None:
            edge_info = self.edge_block(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        return node_info, edge_info

class MiaoNet(AtomicModule):
    """
    Miao nei ga
    duo xi da miao nei
    """
    def __init__(self,
                 embedding_layer : EmbeddingLayer,
                 radial_fn       : RadialLayer,
                 n_layers        : int,
                 max_r_way       : List[int],
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

        max_in_way = [0] + max_out_way[:-1]
        hidden_nodes = [embedding_layer.n_channel] + output_dim
        self.en_equivalent_blocks = self.get_eq_blocks(activate_fn, max_r_way, max_in_way, max_out_way,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers)

        self.readout_layer = ReadoutLayer(n_dim=hidden_nodes[-1],
                                          target_way=target_way,
                                          activate_fn=activate_fn,
                                          bilinear=bilinear,
                                          e_dim=embedding_layer.n_channel)
        # TensorAggregateOP.set_max(max(max_in_way), max(max_out_way), max(max_r_way))

    def calculate(self,
                  batch_data : Dict[str, torch.Tensor],
                  ) -> Dict[str, torch.Tensor]:
        node_info, edge_info = self.get_init_info(batch_data)
        for en_equivalent in self.en_equivalent_blocks:
            node_info, edge_info = en_equivalent(node_info, edge_info, batch_data)
        output_tensors = self.readout_layer(node_info, None)
        if 'site_energy' in output_tensors:
            output_tensors['site_energy'] = output_tensors['site_energy'] * self.std + self.mean
        if 'direct_forces' in output_tensors:
            output_tensors['direct_forces'] = output_tensors['direct_forces'] * self.std
        return output_tensors

    def get_init_info(self,
                      batch_data : Dict[str, torch.Tensor],
                      )->Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        emb = self.embedding_layer(batch_data=batch_data)
        node_info = {0: emb}
        _, dij, _ = find_distances(batch_data)
        rbf = self.radial_fn(dij)
        edge_info = {0: rbf}
        return node_info, edge_info

    def get_eq_blocks(self, activate_fn, max_r_way, max_in_way, max_out_way,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers):
        return nn.ModuleList([
            MiaoBlock(activate_fn=activate_fn,
                      radial_fn=self.radial_fn.replicate(),
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
