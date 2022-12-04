import torch
import numpy as np
from .base import RadialLayer


__all__ = ["ChebyshevPoly",
           "BesselPoly",
           ]


class ChebyshevPoly(RadialLayer):
    def __init__(self, 
                 r_max  : float,
                 r_min  : float=0.5,
                 n_max  : int=8,
                 ) -> None:
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max
        self.n_max = n_max
        n = torch.arange(self.n_max)
        self.register_buffer("n", n)

    def forward(self,
                distances: torch.Tensor,
                ) -> torch.Tensor:
        x = torch.clamp((distances - self.r_min) / (self.r_max - self.r_min),
                        min=0., max=1.0)
        out = torch.cos(torch.arccos(x).unsqueeze(-1) * self.n)
        return out


# DIRECTIONAL MESSAGE PASSING FOR MOLECULAR GRAPHS
class BesselPoly(RadialLayer):
    def __init__(self, 
                 r_max  : float,
                 n_max  : int=8,
                 ) -> None:
        super().__init__()
        self.n_max = n_max
        freqs = torch.arange(1, n_max + 1) * np.pi / r_max
        self.register_buffer("freqs", freqs)

    def forward(self, 
                distances: torch.Tensor,
                ) -> torch.Tensor:
        out = torch.sin(distances.unsqueeze(-1) * self.freqs)
        norm = torch.where(distances == 0, 1.0, distances)
        out = out / norm.unsqueeze(-1)
        return out
