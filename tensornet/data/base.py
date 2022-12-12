import torch
import os
import numpy as np
from tensornet.utils import EnvPara
from torch_geometric.data import Data, InMemoryDataset
from ase.neighborlist import neighbor_list


class AtomsData(InMemoryDataset):
    @staticmethod
    def atoms_to_graph(atoms, cutoff, device='cpu'):
        idx_i, idx_j, offsets = neighbor_list("ijS", atoms, cutoff, self_interaction=False)
        bonds = np.array([idx_i, idx_j])
        offsets = np.array(offsets) @ atoms.get_cell()
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

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, f"processed_{self.cutoff:.2f}".replace(".", "_"))

    @property
    def per_energy_mean(self):
        per_energy = self.data["energy_t"] / self.data["n_atoms"]
        return torch.mean(per_energy)

    @property
    def per_energy_std(self):
        per_energy = self.data["energy_t"] / self.data["n_atoms"]
        return torch.std(per_energy)   

    @property
    def forces_std(self):
        return torch.std(self.data["forces_t"])
    
    @property
    def n_neighbor_mean(self):
        n_neighbor = self.data['edge_index'].shape[1] / len(self.data['x'])
        return n_neighbor
    
    @property
    def all_elements(self):
        return torch.unique(self.data['atomic_number'])

    def load(self, name: str, device: str="cpu") -> None:
        self.data, self.slices = torch.load(name, map_location=device)

    def load_split(self, train_split: str, test_split: str):
        train_idx = np.loadtxt(train_split, dtype=int)
        test_idx  = np.loadtxt(test_split, dtype=int)
        return self.copy(train_idx), self.copy(test_idx)

    def random_split(self, train_num: int, test_num: int):
        assert train_num + test_num < len(self)
        idx = np.random.choice(len(self), train_num + test_num, replace=False)
        train_idx = idx[:train_num]
        test_idx  = idx[train_num:]
        return self.copy(train_idx), self.copy(test_idx)
