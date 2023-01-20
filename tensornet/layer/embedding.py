import torch
from torch import nn
from typing import List, Optional, Dict
from .base import CutoffLayer, EmbeddingLayer
from ..utils import find_distances, _scatter_add

__all__ = ["AtomicOneHot",
           "AtomicNumber",
           "AtomicEmbedding",
           "BehlerG1",
           ]


class AtomicOneHot(EmbeddingLayer):
    def __init__(self, 
                 atomic_number : List[int], 
                 trainable     : bool=False
                 ) -> None:
        super().__init__()
        max_atomic_number = max(atomic_number)
        n_atomic_number = len(atomic_number)
        weights = torch.zeros(max_atomic_number + 1, n_atomic_number)
        for idx, z in enumerate(atomic_number):
            weights[z, idx] = 1.
        self.z_weights = nn.Embedding(max_atomic_number + 1, n_atomic_number)
        self.z_weights.weight.data = weights
        if not trainable:
            self.z_weights.weight.requires_grad = False
        self.n_channel = n_atomic_number

    def forward(self, 
                batch_data : Dict[str, torch.Tensor],
                ) -> torch.Tensor:
        return self.z_weights(batch_data['atomic_number'])


class AtomicNumber(EmbeddingLayer):
    def __init__(self, 
                 atomic_number : List[int], 
                 trainable     : bool=False
                 ) -> None:
        super().__init__()
        max_atomic_number = max(atomic_number)
        weights = torch.arange(max_atomic_number + 1)[:, None].float()
        self.z_weights = nn.Embedding(max_atomic_number + 1, 1)
        self.z_weights.weight.data = weights
        if not trainable:
            self.z_weights.weight.requires_grad = False
        self.n_channel = 1

    def forward(self, 
                batch_data : Dict[str, torch.Tensor],
                ) -> torch.Tensor:
        return self.z_weights(batch_data['atomic_number'])


class AtomicEmbedding(EmbeddingLayer):
    def __init__(self, 
                 atomic_number : List[int], 
                 n_channel     : int,
                 ) -> None:
        super().__init__()
        max_atomic_number = int(max(atomic_number))
        self.z_weights = nn.Embedding(max_atomic_number + 1, n_channel)
        self.n_channel = n_channel

    def forward(self, 
                batch_data : Dict[str, torch.Tensor],
                ) -> torch.Tensor:
        return self.z_weights(batch_data['atomic_number'])


class BehlerG1(EmbeddingLayer):
    """
    wACSF—Weighted atom-centered symmetry functions as descriptors in machine learning potentials
    J. Chem. Phys. 148, 241709 (2018)
    https://doi.org/10.1063/1.5019667
    """
    def __init__(self, 
                 n_radial      : int, 
                 cut_fn        : CutoffLayer, 
                 etas          : Optional[List[float]]=None, 
                 rss           : Optional[List[float]]=None,
                 trainable     : bool=False,
                 ) -> None:
        super().__init__()
        self.cut_fn = cut_fn
        if rss is None or etas is None:
            cutoff = cut_fn.cutoff.numpy()
            rss = torch.linspace(0.3, cutoff - 0.3, n_radial)
            etas = 0.5 * torch.ones_like(rss) / (rss[1] - rss[0]) ** 2
        assert (len(etas) == n_radial) and (len(rss) == n_radial), "Lengths of 'etas' or 'rss' error"

        if trainable:
            self.etas = nn.Parameter(etas)
            self.rss = nn.Parameter(rss)
            # self.etas = PositiveParameter(etas)
        else:
            self.register_buffer("etas", etas)
            self.register_buffer("rss", rss)
        self.n_channel = n_radial

    def forward(self,
                batch_data  : Dict[str, torch.Tensor],
                ) -> torch.Tensor:
        n_atoms = batch_data['atomic_number'].shape[0]
        idx_i, idx_j = batch_data['edge_index']
        _, dij, _ = find_distances(batch_data)
        zij = batch_data['atomic_number'][idx_j].unsqueeze(-1)                      # [n_edge, 1]
        dij = dij.unsqueeze(-1)
        f = torch.exp(-self.etas * (dij - self.rss) ** 2) * self.cut_fn(dij) * zij  # [n_edge, n_channel]
        f = _scatter_add(f, idx_i, dim_size=n_atoms)
        # f = segment_coo(f, idx_i, dim_size=n_atoms, reduce="sum").view(n_atoms, -1)
        return f
