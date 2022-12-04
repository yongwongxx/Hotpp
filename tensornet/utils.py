import torch
import numpy as np
import itertools
import torch.nn.functional as F
from typing import Iterable, Optional, Dict, List


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
                   ) -> None:
    if 'rij' not in batch_data:
        idx_i, idx_j = batch_data.edge_index
        batch_data['rij'] = batch_data['coordinate'][idx_j] + batch_data['offset'] - batch_data['coordinate'][idx_i]
    if 'dij' not in batch_data:
        batch_data['dij'] = torch.norm(batch_data['rij'], dim=-1)
    if 'uij' not in batch_data:
        batch_data['uij'] = batch_data['rij'] / batch_data['dij'].unsqueeze(-1)
    return batch_data['rij'], batch_data['dij'], batch_data['uij']


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


def get_loss(batch_data : Dict[str, torch.Tensor], 
             weight     : List[int]=[1.0, 1.0, 1.0], 
             verbose    : bool=False):
    w_energy, w_forces, w_stress = weight
    loss = 0.
    if w_energy > 0.:
        energy_loss = F.mse_loss(batch_data['energy_p'] / batch_data['n_atoms'], 
                                 batch_data['energy_t'] / batch_data['n_atoms'])
        loss += w_energy * energy_loss
    if w_forces > 0.:
        forces_loss = F.mse_loss(batch_data['forces_p'], batch_data['forces_t'])
        loss += w_forces * forces_loss
    if verbose:
        return loss, energy_loss, forces_loss
    return loss


class EnvPara:
    FLOAT_PRECISION = torch.float


# Steal from our good brother BingqingCheng. Is there a problem in license?
def get_default_acsf_hyperparameters(rmin, cutoff):
    etas, rss = [], []
    N = int((cutoff - rmin) * 3)
    index = np.arange(N + 1, dtype=float)
    shift_array = cutoff * (1. / N) ** (index / (len(index) - 1))
    eta_array = 1. / shift_array ** 2.

    for eta in eta_array:
        # G2 with no shift
        if 3 * np.sqrt(1 / eta) > rmin:
            etas.append(eta)
            rss.append(0.)

    for i in range(len(shift_array)-1):
        # G2 with shift
        eta = 1./((shift_array[N - i] - shift_array[N - i - 1])**2)
        if shift_array[N - i] + 3 * np.sqrt(1 / eta) > rmin:
            etas.append(eta)
            rss.append(shift_array[N-i])
    return etas, rss


def expand_para(para: int or List, n: int):
    assert isinstance(para, int) or isinstance(para, list)
    if isinstance(para, int):
        para = [para] * n
    if isinstance(para, list):
        assert len(para) == n
    return para


@torch.jit.script
def _scatter_add(x        : torch.Tensor, 
                 idx_i    : torch.Tensor, 
                 dim_size : Optional[int]=None, 
                 dim      : int = 0
                 ) -> torch.Tensor:
    shape = list(x.shape)
    if dim_size is None:
        dim_size = idx_i.max() + 1
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y
