import torch
from torch import nn


class RadiusFunction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self,
                distances: torch.Tensor,
                ) -> torch.Tensor:
        raise NotImplementedError()
    

class ChebyshevPoly(RadiusFunction):
    def __init__(self, 
                 r_max  : float,
                 r_min  : float=0.5,
                 n_max  : int=8,
                 ) -> None:
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.n_max = n_max
        self.weights = nn.Parameter(torch.ones(n_max))

    def forward(self,
                distances: torch.Tensor,
                ) -> torch.Tensor:
        out = torch.zeros_like(distances)
        x = (distances - self.r_min) / (self.r_max - self.r_min)
        for n in range(self.n_max):
            out += torch.cos(n * torch.arccos(x))
        out *=  (1 - x) ** 2
        return out

