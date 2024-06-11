import torch
import numpy as np
from .base import RadialLayer, CutoffLayer
from torch import nn
from typing import List, Callable, Optional


__all__ = ["ChebyshevPoly",
           "SpinChebyshevPoly",
           "BesselPoly",
           "MLPPoly"
           ]


class ChebyshevPoly(RadialLayer):
    """
    Atomic cluster expansion for accurate and transferable interatomic potentials    Eq. (24)
    """
    def __init__(self,
                 r_max      : float,
                 r_min      : float=0.5,
                 n_max      : int=8,
                 lamb       : float=5.0,
                 cutoff_fn  : Optional[CutoffLayer]=None,
                 ) -> None:
        super().__init__(n_channel=n_max, cutoff_fn=cutoff_fn)
        self.r_min = r_min
        self.r_max = r_max
        self.n_max = n_max
        self.lamb = lamb
        self.norm = torch.tensor(np.exp(lamb) - 1, dtype=torch.float32)
        n = torch.arange(1, self.n_max + 1)
        self.register_buffer("n", n)

    def radial(self, d: torch.Tensor,) -> torch.Tensor:
        x = 2 * torch.clamp((d - self.r_min) / (self.r_max - self.r_min), min=0.0, max=1.0) - 1
        out = torch.cos(torch.arccos(x) * self.n)
        # x = 2 * (1 - x) ** 2 - 1
        # out = 0.5 * torch.cos(torch.arccos(x) * self.n) + 1
        # x = 1 - 2 * (torch.exp(self.lamb * (1 - x)) - 1) / self.norm
        # out = (1 - torch.cos(torch.arccos(x) * self.n)) * 0.5
        return out

    def replicate(self):
        return self.__class__(self.r_max, self.r_min, self.n_max, self.lamb, self.cutoff_fn)


# DIRECTIONAL MESSAGE PASSING FOR MOLECULAR GRAPHS
class BesselPoly(RadialLayer):
    def __init__(self, 
                 r_max      : float,
                 n_max      : int=8,
                 cutoff_fn  : Optional[CutoffLayer]=None,
                 ) -> None:
        super().__init__(n_channel=n_max, cutoff_fn=cutoff_fn)
        self.n_max = n_max
        self.r_max = r_max
        freqs = torch.arange(1, n_max + 1, dtype=torch.float32) * np.pi / r_max
        freqs.requires_grad_()
        self.freqs = nn.Parameter(freqs)

    def radial(self, d: torch.Tensor) -> torch.Tensor:
        norm = torch.where(d==0.0, 1.0, d) * (self.freqs)
        out = torch.sin(d * self.freqs) / norm
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
        super().__init__(n_channel=n_hidden[-1], cutoff_fn=cutoff_fn)
        self.n_hidden = n_hidden
        self.radial_fn = radial_fn
        self.activate_fn = activate_fn
        x = [nn.Linear(radial_fn.n_channel, n_hidden[0])]
        for n_in, n_out in zip(n_hidden[:-1], n_hidden[1:]):
            x.extend([activate_fn, nn.Linear(n_in, n_out)])
        self.mlp = nn.Sequential(*x)

    def forward(self,
                distances: torch.Tensor,
                ) -> torch.Tensor:
        out = self.mlp(self.radial_fn(distances))
        return out

    def replicate(self):
        return self.__class__(self.n_hidden, self.radial_fn.replicate(), self.activate_fn, self.cutoff_fn)


# class KANPoly(RadialLayer):
#     def __init__(self,
#                  n_hidden     : List[int],
#                  radial_fn    : RadialLayer,
#                  cutoff_fn    : Optional[CutoffLayer]=None,
#                  ) -> None:
#         super().__init__(n_channel=n_hidden[-1] * radial_fn.n_channel, cutoff_fn=cutoff_fn)
#         self.n_hidden = n_hidden
#         self.radial_fn = radial_fn
#         self.kan = nn.ModuleList([nn.Linear(radial_fn.n_channel, n_hidden[0])])
#         for n_in, n_out in zip(n_hidden[:-1], n_hidden[1:]):
#             self.kan.append(nn.Linear(radial_fn.n_channel * n_in, n_out))

#     def forward(self,
#                 distances: torch.Tensor,
#                 ) -> torch.Tensor:
#         out = distances
#         for kan_layer in self.kan:
#             out = kan_layer(self.radial_fn(out))
#         return out

#     def replicate(self):
#         return self.__class__(self.n_hidden, self.radial_fn.replicate(), self.activate_fn, self.cutoff_fn)


class SpinChebyshevPoly(RadialLayer):
    def __init__(self,
                 spin_max   : float,
                 n_max      : int=8,
                 ) -> None:
        super().__init__(n_channel=n_max - 1)
        self.spin_max = spin_max
        self.n_max = n_max
        n = torch.arange(1, self.n_max)
        self.register_buffer("n", n)

    def radial(self, d: torch.Tensor,) -> torch.Tensor:
        x = 1 - 2 * (d.clamp(min=0.0, max=self.spin_max) / self.spin_max) ** 2
        out = (1 - torch.cos(torch.arccos(x) * self.n)) * 0.5
        return out

    def replicate(self):
        return self.__class__(self.spin_max, self.n_max)