import torch
from torch import nn
from typing import List, Dict
from tensornet.utils import expand_to


class ReadoutLayer(nn.Module):
    def __init__(self, 
                 n_dim      : int,
                 target_way : List[int]=[0],
                 ) -> None:
        super().__init__()
        self.readout_list = nn.ModuleList([nn.Linear(n_dim, 1)])
        for way in range(1, max(target_way) + 1):
            self.readout_list.append(nn.Linear(n_dim, 1, bias=False))
        self.target_way = target_way
    
    def forward(self, 
                input_tensors : Dict[int, torch.Tensor],
                atomic_number : torch.Tensor,
                ) -> Dict[int, torch.Tensor]:
        output_tensors = {}
        symbol_mask = atomic_number < 1
        for way in self.target_way:
            input_tensor = torch.transpose(input_tensors[way], 2, -1)
            output_tensor = self.readout_list[way](input_tensor)
            output_tensor = torch.transpose(output_tensor, 2, -1).squeeze(2)  # delete channel dim
            output_tensors[way] = output_tensor.masked_fill(mask=expand_to(symbol_mask, way + 2), 
                                                            value=torch.tensor(0.))
        return output_tensors