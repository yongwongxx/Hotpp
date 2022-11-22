import torch
from torch import nn
from typing import List, Dict


class ReadoutLayer(nn.Module):
    def __init__(self, 
                 n_dim  : int,
                 n_way  : int,
                 ) -> None:
        super().__init__()
        self.read_list = nn.ModuleList([nn.Linear(n_dim, 1)])
        for way in range(1, n_way + 1):
            self.linear_interact_list.append(nn.Linear(n_dim, 1, bias=False))
    
    def forward(self, 
                input_tensors : Dict[int: torch.Tensor],
                ) -> Dict[int: torch.Tensor]:
        output_tensors = {}
        for way in input_tensors:
            input_tensor = torch.transpose(input_tensors[way], 2, -1)
            output_tensor = self.linear_interact_list[way](input_tensor)
            output_tensors[way] = torch.transpose(output_tensor, 2, -1)
        return output_tensors