import torch
from torch import nn
import numpy as np


class CutoffLayer(nn.Module):
    def __init__(self,
                 cutoff : float=3.5,
                 ) -> None:
        super().__init__()
        self.register_buffer("cutoff", torch.tensor(cutoff).float())
        
    def forward(self,
                ):
        raise NotImplementedError(f"{self.__class__.__name__} must have 'forward'!")


class CosineCutoff(CutoffLayer):
    def forward(self, 
                distances : torch.Tensor,
                ) -> torch.Tensor:
        cutoffs = 0.5 * (torch.cos(distances * np.pi / self.cutoff) + 1.0)
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


class SmoothCosineCutoff(CutoffLayer):
    def __init__(self, 
                 cutoff_smooth : float=2.8, 
                 cutoff        : float=3.5
                 ) -> None:
        assert cutoff_smooth < cutoff, "'cutoff_smooth must smaller than cutoff"
        super(SmoothCosineCutoff, self).__init__(cutoff)
        self.register_buffer("cutoff_smooth", torch.tensor(cutoff_smooth).float())

    def forward(self, 
                distances : torch.Tensor,
                ) -> torch.Tensor:
        phase = (distances.clamp(min=self.cutoff_smooth, max=self.cutoff) - self.cutoff_smooth)/\
                 (self.cutoff - self.cutoff_smooth) * np.pi
        cutoffs = 0.5 * (torch.cos(phase) + 1.0) / (distances + 0.01)
        return cutoffs


class PolynomialCutoff(CutoffLayer):
    def forward(self, 
                distances : torch.Tensor,
                ) -> torch.Tensor:
        cutoffs = (1.0 - (distances / self.cutoff) ** 2) ** 3
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs
