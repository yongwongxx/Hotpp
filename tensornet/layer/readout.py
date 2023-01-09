import torch
from torch import nn
from typing import List, Dict, Callable
from .equivalent import TensorLinear
from .activate import TensorActivateDict


__all__ = ["ReadoutLayer"]


class ReadoutLayer(nn.Module):
    def __init__(self, 
                 n_dim       : int,
                 target_way  : Dict[int, str]={0: "site_energy"},
                 activate_fn : str="jilu",
                 ) -> None:
        super().__init__()
        self.target_way = target_way
        self.layer_dict = nn.ModuleDict({
            prop: nn.Sequential(
                TensorLinear(n_dim, n_dim, bias=(way==0)),
                TensorActivateDict[activate_fn](n_dim),
                TensorLinear(n_dim, 1, bias=(way==0)),
                )
            for way, prop in target_way.items()
            })

    def forward(self, 
                batch_data : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = {}
        for way, prop in self.target_way.items():
            output_tensors[prop] = self.layer_dict[prop](batch_data['node_attr'][way]).squeeze(1)  # delete channel dim
        return output_tensors
