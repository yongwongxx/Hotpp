import torch
import numpy as np
import itertools
from einops import rearrange, reduce, repeat


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def way_combination(max_out, max_in, max_r):
    for o, i, r in itertools.product(range(max_out), range(max_in), range(max_r)):
        z = (i + r - o) / 2
        if 0 <= z <= min(i, r) and int(z) == z:
            yield (o, i, r)


def expand_to(t, n_dim, dim=-1):
    """Expand dimension of the input tensor t at location 'dim' until the total dimention arrive 'n_dim'

    Args:
        t (torch.Tensor): Tensor to expand
        n_dim (int): target dimension 
        dim (int, optional): location to insert axis. Defaults to -1.

    Returns:
        torch.Tensor: Expanded Tensor
    """
    while len(t.shape) < n_dim:
        t = torch.unsqueeze(t, dim=dim)
    return t


def multi_outer_product(v: torch.TensorType, n: int) -> torch.Tensor:
    """Calculate 'n' times outer product of vector 'v'

    Args:
        v (torch.TensorType): vector or vectors ([n_dim] or [..., n_dim])
        n (int): outer prodcut times, will return [...] 1 if n = 0

    Returns:
        torch.Tensor: OvO
    """
    out = torch.ones_like(v[..., 0])
    for _ in range(n):
        out = out[..., None] * expand_to(v, len(out.shape) + 1, dim=len(v.shape) - 1)
    return out


def find_distances(coordinate: torch.Tensor) -> torch.Tensor:
    """get distances between atoms

    Args:
        coordinate (torch.Tensor): coordinate of atoms [n_batch, n_atoms, n_dim]

    Returns:
        torch.Tensor: distances [n_batch, n_atoms, n_atoms, n_dim]
    """
    ri = rearrange(coordinate, 'b i d -> b i () d')
    rj = rearrange(coordinate, 'b j d -> b () j d')
    rij = ri - rj
    return rij
