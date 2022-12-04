# TODO 
# activate_fn should be changed to tensor activate_fn
import torch
from torch import nn
from typing import List, Dict, Callable


__all__ = ["ReadoutLayer"]
class ReadoutLayer(nn.Module):
    def __init__(self, 
                 n_dim       : int,
                 target_way  : List[int]=[0],
                 activate_fn : Callable=torch.relu,
                 ) -> None:
        super().__init__()
        self.layer1_list = nn.ModuleList([nn.Linear(n_dim, n_dim)])
        for way in range(1, max(target_way) + 1):
            self.layer1_list.append(nn.Linear(n_dim, n_dim, bias=False))
        self.layer2_list = nn.ModuleList([nn.Linear(n_dim, 1)])
        for way in range(1, max(target_way) + 1):
            self.layer2_list.append(nn.Linear(n_dim, 1, bias=False))
        self.target_way = target_way
        self.activate_fn = activate_fn
    
    def forward(self, 
                batch_data : Dict[str, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = {}
        for way in self.target_way:
            input_tensor  = torch.transpose(batch_data['node_attr'][way], 1, -1)
            output_tensor = self.layer1_list[way](input_tensor)
            output_tensor = self.activate_fn(output_tensor)
            output_tensor = self.layer2_list[way](output_tensor)
            output_tensor = torch.transpose(output_tensor, 1, -1).squeeze(1)  # delete channel dim
            output_tensors[way] = output_tensor
        return output_tensors
