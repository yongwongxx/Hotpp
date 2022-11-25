import torch
import numpy as np
import itertools
from einops import rearrange, reduce, repeat
from typing import Iterable, Optional, Dict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

def way_combination(out_way : Iterable, 
                    in_way  : Iterable, 
                    r_way   : Iterable
                    ) -> Iterable:
    for o, i, r in itertools.product(out_way, in_way, r_way):
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


def multi_outer_product(v: torch.Tensor, 
                        n: int) -> torch.Tensor:
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


def find_distances(batch_data : Dict[str, torch.Tensor],
                #    coordinate : torch.Tensor,
                #    neighbor   : torch.Tensor,
                #    mask       : torch.Tensor,
                #    cell       : Optional[torch.Tensor]=None,
                #    offset     : Optional[torch.Tensor]=None,
                   ) -> None:
    """get distances between atoms

    Args:
        coordinate (torch.Tensor): coordinate of atoms [n_batch, n_atoms, n_dim]  (float)
        neighbor (torch.Tensor): neighbor of atoms [n_batch, n_atoms, n_neigh]    (int)
        cell (torch.Tensor): cell of atoms [n_batch, n_atoms, n_dim, n_dim]       (float)
        offset (torch.Tensor): offset of cells [n_batch, n_atoms, n_neigh, n_dim] (int)

    Returns:
        torch.Tensor: distances [n_batch, n_atoms, n_neigh, n_dim]
    """
    if 'rij' not in batch_data:
        coordinate = batch_data['coordinate']
        neighbor   = batch_data['neighbor']
        mask       = batch_data['mask']
        offset     = batch_data['offset']
        n_batch = neighbor.shape[0]

        # TODO: which is faster?
        # ri = repeat(coordinate, 'b i d -> b i j d', j=n_neigh)
        # rj = repeat(coordinate, 'b j d -> b i j d', i=n_atoms).gather(2, repeat(neighbor, 'b i j -> b i j d', d=n_dim))

        idx_m = torch.arange(n_batch, device=coordinate.device)[:, None, None]
        ri = coordinate[:, :, None, :]
        rj = coordinate[idx_m, neighbor]
        if offset is not None:
            rj += offset
        distances = rj - ri
        mask = torch.unsqueeze(mask < 0.5, dim=-1)
        distances = distances.masked_fill(mask=mask, value=torch.tensor(0.))
        batch_data['rij'] = distances
    if 'dij' not in batch_data:
        batch_data['dij'] = torch.norm(batch_data['rij'], dim=-1)
    return None


def get_elements(frames):
    elements = set()
    for atoms in frames:
        elements.update(set(atoms.get_atomic_numbers()))
    return list(elements)


def get_mindist(frames):
    min_dist = 100
    for atoms in frames:
        distances = atoms.get_all_distances(mic=True) + np.eye(len(atoms)) * 100
        min_dist = min(np.min(distances), min_dist)
    return min_dist


# TODO: incremental Welford algorithm?
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def translate_energy(frames):
    energy_peratom = []
    for atoms in frames:
        energy_peratom.append(atoms.info['energy'] / len(atoms))
    mean = np.mean(energy_peratom)
    for atoms in frames:
        atoms.info['energy'] -= mean * len(atoms)
    return frames