from typing import Callable, List, Dict, Optional, Literal
import torch
from torch import nn
from .base import AtomicModule
from ..layer import EmbeddingLayer, RadialLayer, ReadoutLayer
from ..layer.equivalent import MultiBodyLayer, GraphConvLayer, NonLinearLayer, GraphNorm
from ..utils import find_distances, _scatter_add, res_add
from .miao import MiaoNet


#TODO graph norm
class UpdateNodeBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_n_body     : int,
                 max_r_way      : int,
                 max_in_way     : int,
                 max_out_way    : int,
                 input_dim      : int,
                 output_dim     : int,
                 norm_factor    : float=1.0,
                 norm           : str='none',
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
        self.self_interact = MultiBodyLayer(max_n_body=max_n_body,
                                            input_dim=input_dim, 
                                            output_dim=output_dim,
                                            max_way=max_out_way)
        self.non_linear = NonLinearLayer(activate_fn=activate_fn,
                                         max_way=max_out_way,
                                         input_dim=output_dim)
        self.register_buffer("norm_factor", torch.tensor(norm_factor))

        self.norm = None
        if norm == 'graph':
            self.norm = GraphNorm(max_way=max_out_way, n_channel=output_dim)
            

    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        message = self.graph_conv(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        res_info = {}
        idx_i = batch_data["idx_i"]
        n_atoms = batch_data['atomic_number'].shape[0]
        for way in message.keys():
            res_info[way] = _scatter_add(message[way], idx_i, dim_size=n_atoms) / self.norm_factor
        res_info = self.non_linear(self.self_interact(res_info))
        if self.norm is not None:
            res_info = self.norm(res_info, batch_data['batch'], batch_data['n_atoms'])
        return res_add(node_info, res_info)


class UpdateEdgeBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_n_body     : int,
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
        self.self_interact = MultiBodyLayer(max_n_body=max_n_body,
                                            input_dim=input_dim,
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


class MiaoMiaoBlock(nn.Module):
    def __init__(self,
                 radial_fn      : RadialLayer,
                 max_n_body     : int,
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
                                          max_n_body=max_n_body,
                                          max_r_way=max_r_way, 
                                          max_in_way=max_in_way,
                                          max_out_way=max_out_way,
                                          input_dim=input_dim, 
                                          output_dim=output_dim, 
                                          norm_factor=norm_factor, 
                                          activate_fn=activate_fn, 
                                          conv_mode=conv_mode)
        if update_edge:
            self.edge_block = UpdateEdgeBlock(radial_fn=radial_fn, 
                                            max_n_body=max_n_body,
                                            max_r_way=max_r_way, 
                                            max_in_way=max_in_way,
                                            max_out_way=max_out_way,
                                            input_dim=input_dim,
                                            output_dim=output_dim,
                                            activate_fn=activate_fn,
                                            conv_mode=conv_mode)
        self.update_edge = update_edge


    def forward(self,
                node_info    : Dict[int, torch.Tensor],
                edge_info    : Dict[int, torch.Tensor],
                batch_data   : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        node_info = self.node_block(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        if self.update_edge:
            edge_info = self.edge_block(node_info=node_info, edge_info=edge_info, batch_data=batch_data)
        return node_info, edge_info

class MiaoMiaoNet(MiaoNet):
    def __init__(self,
                 embedding_layer : EmbeddingLayer,
                 radial_fn       : RadialLayer,
                 n_layers        : int,
                 max_n_body      : List[int],
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
        self.max_n_body = max_n_body
        super().__init__(embedding_layer, radial_fn, n_layers, max_r_way, max_out_way, output_dim, activate_fn, target_way, mean, std, norm_factor, bilinear, conv_mode, update_edge)
        
    def get_eq_blocks(self, activate_fn, max_r_way, max_in_way, max_out_way,
            hidden_nodes, norm_factor, conv_mode, update_edge, n_layers):
        return nn.ModuleList([
            MiaoMiaoBlock(activate_fn=activate_fn,
                          radial_fn=self.radial_fn.replicate(),
                          # Use factory method, so the radial_fn in each layer are different
                          max_n_body=self.max_n_body[i],
                          max_r_way=max_r_way[i],
                          max_in_way=max_in_way[i],
                          max_out_way=max_out_way[i],
                          input_dim=hidden_nodes[i],
                          output_dim=hidden_nodes[i + 1],
                          norm_factor=norm_factor,
                          conv_mode=conv_mode,
                          update_edge=update_edge,
                          ) for i in range(n_layers)])
