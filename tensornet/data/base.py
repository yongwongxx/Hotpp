import torch
import numpy as np
from tensornet.utils import EnvPara
from torch_geometric.data import Data, InMemoryDataset
from ase.neighborlist import neighbor_list


__all__ = ["atoms_to_graph",
           "AtomsData",
           "PtData",
           ]

def atoms_to_graph(atoms, cutoff, device='cpu'):
    idx_i, idx_j, offsets = neighbor_list("ijS", atoms, cutoff, self_interaction=False)
    bonds = np.array([idx_i, idx_j])
    offsets = np.array(offsets, dtype=np.float32) @ atoms.get_cell()
    index = torch.arange(len(atoms), dtype=torch.long, device=device)
    atomic_number = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
    edge_index = torch.tensor(bonds, dtype=torch.long, device=device)
    offset = torch.tensor(offsets, dtype=EnvPara.FLOAT_PRECISION, device=device)
    coordinate = torch.tensor(atoms.positions, dtype=EnvPara.FLOAT_PRECISION, device=device)
    n_atoms = torch.tensor(len(atoms), dtype=EnvPara.FLOAT_PRECISION, device=device)
    graph = Data(x=index,
                 atomic_number=atomic_number, 
                 edge_index=edge_index,
                 offset=offset,
                 coordinate=coordinate,
                 n_atoms=n_atoms,
                 )
    for key in ['site_energy', 'energy', 'forces']:#, 'stress']:
        graph['has_' + key] = False
        if key in atoms.info:
            graph[key + '_t'] = torch.tensor(atoms.info[key], dtype=EnvPara.FLOAT_PRECISION, device=device)
            graph['has_' + key] = True
    return graph


class AtomsData(InMemoryDataset):
    def __init__(self, frames, cutoff, device='cpu'):
        super().__init__()
        self.frames = frames
        self.cutoff = cutoff
        self.datalist = [{} for _ in range(len(frames))]
        self.device = device

    def len(self):
        return len(self.frames)

    def extend(self, frames):
        self.frames.extend(frames)
        self.datalist.extend([{} for _ in range(len(frames))])

    def get(self, i):
        if not self.datalist[i]:
            self.datalist[i] = atoms_to_graph(self.frames[i], self.cutoff, self.device)
        return self.datalist[i]

    def remove(self, index):
        for i in sorted(index, reverse=True):
            self.frames.pop(i)
            self.datalist.pop(i)


class PtData(InMemoryDataset):
    def __init__(self, name: str, device: str="cpu") -> None:
        super().__init__()
        self.data, self.slices = torch.load(name, map_location=device)
