import torch
from torch import nn
from einops import rearrange
from typing import List, Optional, Dict
from .cutoff import CutoffLayer
from ..utils import find_distances


class EmbeddingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                coordinate    : torch.Tensor,
                cell          : torch.Tensor,
                atomic_number : torch.Tensor,
                ) -> torch.Tensor:
        """

        Args:
            coordinate (torch.Tensor): coordinate of atoms [n_batch, n_atoms, n_dim]
            cell (torch.Tensor): cell of atoms [n_batch, n_atoms, n_dim, n_dim]
            atomic_number (torch.Tensor): atomic numbers of atoms [n_batch, n_atoms]

        Returns:
            torch.Tensor: Embedding of atoms [n_batch, n_atoms, n_channel]
        """
        raise NotImplementedError(f"{self.__class__.__name__} must have 'forward'!")


class OneHot(nn.Module):
    def __init__(self, 
                 atomic_number : List[int], 
                 trainable     : bool=False
                 ) -> None:
        super(OneHot, self).__init__()
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
                atomic_number : torch.Tensor
                ) -> torch.Tensor:
        return self.z_weights(atomic_number)


class AtomicNumber(nn.Module):
    def __init__(self, 
                 atomic_number : List[int], 
                 trainable     : bool=False
                 ) -> None:
        super(AtomicNumber, self).__init__()
        max_atomic_number = max(atomic_number)
        weights = torch.arange(max_atomic_number + 1)[:, None].float()
        self.z_weights = nn.Embedding(max_atomic_number + 1, 1)
        self.z_weights.weight.data = weights
        if not trainable:
            self.z_weights.weight.requires_grad = False
        self.n_channel = 1

    def forward(self, 
                atomic_number : torch.Tensor
                ) -> torch.Tensor:
        return self.z_weights(atomic_number)


class BehlerG1(EmbeddingLayer):
    def __init__(self, 
                 n_radius      : int, 
                 cut_fn        : CutoffLayer, 
                 atomic_fn     : EmbeddingLayer,
                 etas          : Optional[List[float]]=None, 
                 rss           : Optional[List[float]]=None,
                 trainable     : bool=False,
                 ) -> None:
        super(BehlerG1, self).__init__()
        self.cut_fn = cut_fn
        self.atomic_embedding = atomic_fn

        if rss is None or etas is None:
            cutoff = cut_fn.cutoff.numpy()
            rss = torch.linspace(0.3, cutoff - 0.3, n_radius)
            etas = 0.5 * torch.ones_like(rss) / (rss[1] - rss[0]) ** 2

        assert (len(etas) == n_radius) and (len(rss) == n_radius), "Lengths of 'etas' or 'rss' error"

        if trainable:
            self.etas = nn.Parameter(etas)
            self.rss = nn.Parameter(rss)
            # self.etas = PositiveParameter(etas)
        else:
            self.register_buffer("etas", etas)
            self.register_buffer("rss", rss)
        
        self.n_channel = n_radius * self.atomic_embedding.n_channel

    def forward(self,
                batch_data  : Dict[str, torch.Tensor],
                # coordinate    : torch.Tensor,
                # atomic_number : torch.Tensor,
                # neighbor      : torch.Tensor,
                # mask          : torch.Tensor,
                # cell          : Optional[torch.Tensor]=None,
                # offset        : Optional[torch.Tensor]=None,
                ) -> torch.Tensor:
        atomic_number = batch_data['atomic_number']
        neighbor = batch_data['neighbor']
        n_batch, n_atoms = atomic_number.shape
        device = atomic_number.device

        z_ratio = self.atomic_embedding(atomic_number)
        idx_m = torch.arange(n_batch, device=device)[:, None, None]
        z_ij = z_ratio[idx_m, neighbor]

        find_distances(batch_data)
        d_ij = batch_data['dij'].unsqueeze(-1)
        x = -self.etas * (d_ij - self.rss) ** 2
        cut = self.cut_fn(d_ij)
        f = torch.exp(x) * cut
        # f: (n_batch, n_atoms, n_neigh, n_channel)
        # z_ij: (n_batch, n_atoms, n_neigh, n_embedded)
        f = torch.sum(f.unsqueeze(-1) * z_ij.unsqueeze(-2), dim=2).view(n_batch, n_atoms, -1)
        # f = rearrange(torch.einsum('b i j c, b i j e -> b i c e', f, z_ij), 'b i c e -> b i (c e)')
        return f
