# TODO 
# activate_fn should be changed to tensor activate_fn
import torch
from torch import nn
from typing import List, Dict, Callable
from .equivalent import TensorDense


__all__ = ["ReadoutLayer"]


class ReadoutLayer(nn.Module):
    def __init__(self, 
                 n_dim       : int,
                 target_way  : Dict[int, str]={0: "site_energy"},
                 activate_fn : Callable=torch.relu,
                 ) -> None:
        super().__init__()
        self.target_way = target_way
        self.layer_dict = nn.ModuleDict({
            prop: nn.Sequential(
                TensorDense(n_dim, n_dim, activate_fn=activate_fn),
                TensorDense(n_dim, 1, activate_fn=None),
                )
            for prop in target_way.values()
            })

    def forward(self, 
                batch_data : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = {}
        for way, prop in self.target_way.items():
            output_tensors[prop] = self.layer_dict[prop](batch_data['node_attr'][way]).squeeze(1)  # delete channel dim
        return output_tensors
