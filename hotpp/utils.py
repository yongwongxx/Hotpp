import sys
import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Dict, List, Callable, Tuple, Union


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def expand_to(t     : torch.Tensor,
              n_dim : int,
              dim   : int=-1) -> torch.Tensor:
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


def add_scaling(batch_data  : Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:
    if 'has_add_scaling' not in batch_data:
        idx_m = batch_data['batch']
        idx_i = batch_data['idx_i']
        batch_data['coordinate'] = torch.matmul(batch_data['coordinate'][:, None, :], 
                                                batch_data['scaling'][idx_m]).squeeze(1)
        if 'offset' in batch_data:
            batch_data['offset'] = torch.matmul(batch_data['offset'][:, None, :],
                                                batch_data['scaling'][idx_m][idx_i]).squeeze(1)
        batch_data['has_add_scaling'] = torch.tensor(True)
    return batch_data


def find_distances(batch_data  : Dict[str, torch.Tensor],) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if 'rij' not in batch_data:
        idx_i = batch_data["idx_i"]
        if 'ghost_neigh' in batch_data:                  # neighbor for lammps calculation
            idx_j = batch_data["ghost_neigh"]
        else:
            idx_j = batch_data["idx_j"]
        if 'offset' in batch_data:
            batch_data['rij'] = batch_data['coordinate'][idx_j] + batch_data['offset'] - batch_data['coordinate'][idx_i]
        else:
            batch_data['rij'] = batch_data['coordinate'][idx_j] - batch_data['coordinate'][idx_i]
    if 'dij' not in batch_data:
        batch_data['dij'] = torch.norm(batch_data['rij'], dim=-1)
    if 'uij' not in batch_data:
        batch_data['uij'] = batch_data['rij'] / batch_data['dij'].unsqueeze(-1)
    return batch_data['rij'], batch_data['dij'], batch_data['uij']


def find_spin(batch_data  : Dict[str, torch.Tensor],) -> Tuple[torch.Tensor]:
    if 'mi' not in batch_data:
        batch_data['mi'] = torch.norm(batch_data['spin'], dim=-1)
    if 'si' not in batch_data:
        norm = torch.where(batch_data['mi'] == 0.0, 1.0, batch_data['mi'])
        batch_data['si'] = batch_data['spin'] / norm.unsqueeze(-1)
    if 'sij' not in batch_data:
        batch_data['sij'] = torch.sum(
            batch_data['si'][batch_data['idx_i']] * batch_data['si'][batch_data['idx_j']], dim=1)
    return batch_data['mi'], batch_data['si'], batch_data['sij']


def find_moment(batch_data  : Dict[str, torch.Tensor],
                n_way       : int
                ) -> torch.Tensor:
    if 'moment' + str(n_way) not in batch_data:
        find_distances(batch_data)
        batch_data['moment' + str(n_way)] = multi_outer_product(batch_data['uij'], n_way)
    return batch_data['moment' + str(n_way)]


def find_spin_moment(batch_data  : Dict[str, torch.Tensor],
                     n_way       : int
                     ) -> torch.Tensor:
    if 'spin_moment' + str(n_way) not in batch_data:
        batch_data['spin_moment' + str(n_way)] = multi_outer_product(batch_data['si'], n_way)
    return batch_data['spin_moment' + str(n_way)]

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


def expand_para(para: Union[int, List[int]], n: int):
    assert isinstance(para, int) or isinstance(para, list)
    if isinstance(para, int):
        para = [para] * n
    if isinstance(para, list):
        assert len(para) == n
    return para


def res_add(t1: Dict[int, torch.Tensor], 
            t2: Dict[int, torch.Tensor]
            ) -> Dict[int, torch.Tensor]:
    for k in t2:
        if k in t1:
            t1[k] = t1[k] + t2[k]
        else:
            t1[k] = t2[k]
    return t1


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


def progress_bar(i: int, n: int, interval: int=100):
    if i % interval == 0:
        ii = int(i / n * 100)
        print(f"\r{ii}%[{'*' * ii}{'-' * (100 - ii)}]", end=' ', file=sys.stderr)


@torch.jit.script
def _aggregate(moment_tensor: torch.Tensor,
               fn : torch.Tensor,
               input_tensor:torch.Tensor,
               in_way : int,
               r_way : int,
               out_way: int
               ) -> torch.Tensor:
    filter_tensor = moment_tensor.unsqueeze(1) * expand_to(fn, n_dim=r_way + 2) # [n_edge, n_channel, n_dim, n_dim, ...]
    coupling_way = (in_way + r_way - out_way) // 2
    n_way = in_way + r_way - coupling_way + 2
    input_tensor  = expand_to(input_tensor, n_way, dim=-1)
    filter_tensor = expand_to(filter_tensor, n_way, dim=2)
    output_tensor = input_tensor * filter_tensor
    # input_tensor:  [n_edge, n_channel, n_dim, n_dim, ...,     1] 
    # filter_tensor: [n_edge, n_channel,     1,     1, ..., n_dim]  
    # with (in_way + r_way - coupling_way) dim after n_channel
    # We should sum up (coupling_way) n_dim
    if coupling_way > 0:
        sum_axis = [i for i in range(in_way - coupling_way + 2, in_way + 2)]
        output_tensor = torch.sum(output_tensor, dim=sum_axis)
    return output_tensor


@torch.jit.script
def _aggregate_new(T1: torch.Tensor,
                   T2: torch.Tensor,
                   way1 : int,
                   way2 : int,
                   way3 : int,
                   ) -> torch.Tensor:
    coupling_way = (way1 + way2 - way3) // 2
    n_way = way1 + way2 - coupling_way + 2
    output_tensor = expand_to(T1, n_way, dim=-1) * expand_to(T2, n_way, dim=2)
    # T1:  [n_edge, n_channel, n_dim, n_dim, ...,     1] 
    # T2:  [n_edge, n_channel,     1,     1, ..., n_dim]  
    # with (way1 + way2 - coupling_way) dim after n_channel
    # We should sum up (coupling_way) n_dim
    if coupling_way > 0:
        sum_axis = [i for i in range(way1 - coupling_way + 2, way1 + 2)]
        output_tensor = torch.sum(output_tensor, dim=sum_axis)
    return output_tensor

        
def aggregate_tensors_fn(way1 : int,
                         way2 : int,
                         way3 : int,
                         ) -> Callable:
    coupling_way = (way1 + way2 - way3) // 2
    n_way = way1 + way2 - coupling_way + 2
    sum_axis = [i for i in range(way1 - coupling_way + 2, way1 + 2)]
    def aggregate_tensors(T1: torch.Tensor,
                          T2: torch.Tensor,
                          ) -> torch.Tensor:
        output_tensor = expand_to(T1, n_way, dim=-1) * expand_to(T2, n_way, dim=2)
        if coupling_way > 0:
            output_tensor = torch.sum(output_tensor, dim=sum_axis)
        return output_tensor
    # return torch.compile(aggregate_tensors)
    return torch.jit.script(aggregate_tensors)


class TensorAggregateOP:
    oplist = {}
    @classmethod
    def set_max(cls, max_way1, max_way2, max_way3):
        for way1 in range(max_way1 + 1):
            for way2 in range(max_way2 + 1):
                for way3 in range(abs(way2 - way1), min(max_way3, way1 + way2) + 1, 2):
                    cls.oplist[(way1, way2, way3)] = aggregate_tensors_fn(way1, way2, way3)
