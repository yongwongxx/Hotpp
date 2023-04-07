import torch
import numpy as np
from .base import RadialLayer, CutoffLayer
from torch import nn
from typing import List, Callable, Optional


__all__ = ["ChebyshevPoly",
           "BesselPoly",
           "MLPPoly"
           ]


class ChebyshevPoly(RadialLayer):
    def __init__(self, 
                 r_max      : float,
                 r_min      : float=0.5,
                 n_max      : int=8,
                 cutoff_fn  : Optional[CutoffLayer]=None,
                 ) -> None:
        super().__init__(n_features=n_max, cutoff_fn=cutoff_fn)
        self.r_min = r_min
        self.r_max = r_max
        self.n_max = n_max
        n = torch.arange(self.n_max)
        self.register_buffer("n", n)

    def radial(self, d: torch.Tensor,) -> torch.Tensor:
        x = torch.clamp((d - self.r_min) / (self.r_max - self.r_min), min=0., max=1.0)
        out = torch.cos(torch.arccos(x) * self.n)
        return out

    def replicate(self):
        return self.__class__(self.r_max, self.r_min, self.n_max, self.cutoff_fn)

# DIRECTIONAL MESSAGE PASSING FOR MOLECULAR GRAPHS
class BesselPoly(RadialLayer):
    def __init__(self, 
                 r_max      : float,
                 n_max      : int=8,
                 cutoff_fn  : Optional[CutoffLayer]=None,
                 ) -> None:
        super().__init__(n_features=n_max, cutoff_fn=cutoff_fn)
        self.n_max = n_max
        self.r_max = r_max
        freqs = torch.arange(1, n_max + 1, dtype=torch.float32) * np.pi / r_max
        freqs.requires_grad_()
        self.freqs = nn.Parameter(freqs)

    def radial(self, d: torch.Tensor) -> torch.Tensor:
        out = torch.sin(d * self.freqs) / d
        return out

    def replicate(self):
        return self.__class__(self.r_max, self.n_max, self.cutoff_fn)

class MLPPoly(RadialLayer):
    def __init__(self,
                 n_hidden     : List,
                 radial_fn    : RadialLayer,
                 activate_fn  : nn.Module=nn.SiLU(),
                 cutoff_fn    : Optional[CutoffLayer]=None,
                 ) -> None:
        super().__init__(n_features=n_hidden[-1], cutoff_fn=cutoff_fn)
        self.n_hidden = n_hidden
        self.radial_fn = radial_fn
        self.activate_fn = activate_fn
        x = [nn.Linear(radial_fn.n_features, n_hidden[0], bias=False)]
        for n_in, n_out in zip(n_hidden[:-1], n_hidden[1:]):
            x.extend([activate_fn, nn.Linear(n_in, n_out, bias=False)])
        self.mlp = nn.Sequential(*x)

    def forward(self,
                distances: torch.Tensor,
                ) -> torch.Tensor:
        out = self.mlp(self.radial_fn(distances))
        return out

    def replicate(self):
        return self.__class__(self.n_hidden, self.radial_fn.replicate(), self.activate_fn, self.cutoff_fn)
