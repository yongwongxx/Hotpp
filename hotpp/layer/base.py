import torch
from torch import nn
from typing import Dict, Optional


__all__ = ["RadialLayer",
           "EmbeddingLayer",
           "CutoffLayer",
           "TensorActivateLayer",
           ]
   

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
                 cutoff : float,
                 ) -> None:
        super().__init__()
        self.register_buffer("cutoff", torch.tensor(cutoff).float())
        
    def forward(self,
                ):
        raise NotImplementedError(f"{self.__class__.__name__} must have 'forward'!")


class RadialLayer(nn.Module):
    def __init__(self, 
                 n_channel : int,
                 cutoff_fn  : Optional[CutoffLayer]=None) -> None:
        super().__init__()
        self.n_channel = n_channel
        self.cutoff_fn = cutoff_fn

    def radial(self,
               d: torch.Tensor,  # [n, 1]
               ) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} must have 'radial'!")

    def forward(self,
                distances: torch.Tensor,
                ) -> torch.Tensor:
        d = distances.unsqueeze(-1)
        if self.cutoff_fn is None:
            return self.radial(d)
        else:
            return self.radial(d) * self.cutoff_fn(d)

    def replicate(self):
        raise NotImplementedError(f"{self.__class__.__name__} must have 'replicate'!")


class TensorActivateLayer(nn.Module):
    """Activate function for tensor inputs with shape [n_batch, n_channel, n_dim, n_dim, ...]
    For tensor with way more than one, bias should be different so the number of input_dim is required.
    """
    def __init__(self,
                 input_dim : int,
                 ) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(input_dim, requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(input_dim, requires_grad=True))

    def forward(self,
                input_tensor: torch.Tensor,
                ) -> torch.Tensor:
        way = len(input_tensor.shape) - 2
        if way == 0:
            return self.activate(input_tensor)
        else:
            return self.tensor_activate(input_tensor, way=way)

    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} must have 'activate'!")

    def tensor_activate(self,
                        input_tensor: torch.Tensor,
                        way         : int,
                        ) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} must have 'tensor_activate'!")
