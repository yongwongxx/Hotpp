import torch
from torch import nn
from typing import Dict


__all__ = ["RadialLayer",
           "EmbeddingLayer",
           "CutoffLayer",
           ]


class RadialLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self,
                distances: torch.Tensor,
                ) -> torch.Tensor:
        raise NotImplementedError()
    

class EmbeddingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                batch_data : Dict[str, torch.Tensor],
                ) -> torch.Tensor:
        """
        Args:
            batch_data  : Tensor dict
        Returns:
            torch.Tensor: Embedding of atoms [n_atoms, n_channel]
        """
        raise NotImplementedError(f"{self.__class__.__name__} must have 'forward'!")


class CutoffLayer(nn.Module):
    def __init__(self,
                 cutoff : float=3.5,
                 ) -> None:
        super().__init__()
        self.register_buffer("cutoff", torch.tensor(cutoff).float())
        
    def forward(self,
                ):
        raise NotImplementedError(f"{self.__class__.__name__} must have 'forward'!")

