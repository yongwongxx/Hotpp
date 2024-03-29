import torch
import copy
import abc
import numpy as np
from ..utils import EnvPara
from torch.utils.data import Dataset
from ase.neighborlist import neighbor_list
from typing import Optional, List

# TODO: offset and scaling for different condition
class AtomsDataset(Dataset, abc.ABC):

    @staticmethod
    def atoms_to_data(atoms, cutoff, properties=['energy', 'forces']):
        dim = len(atoms.get_cell())
        idx_i, idx_j, offset = neighbor_list("ijS", atoms, cutoff, self_interaction=False)
        offset = np.array(offset) @ atoms.get_cell()

        data = {
            "atomic_number": torch.tensor(atoms.numbers, dtype=torch.long),
            "idx_i": torch.tensor(idx_i, dtype=torch.long),
            "idx_j": torch.tensor(idx_j, dtype=torch.long),
            "coordinate": torch.tensor(atoms.positions, dtype=EnvPara.FLOAT_PRECISION),
            "n_atoms": torch.tensor([len(atoms)], dtype=torch.long),
            "offset": torch.tensor(offset, dtype=EnvPara.FLOAT_PRECISION),
        }
        if 'energy' in properties:
            if 'energy' not in atoms.info:
                try:
                    atoms.info['energy'] = atoms.get_potential_energy()
                except:
                    pass
        if 'forces' in properties:
            if 'forces' not in atoms.info:
                try:
                    atoms.info['forces'] = atoms.get_forces()
                except:
                    pass
        if 'virial' in properties:
            data["scaling"] = torch.eye(dim, dtype=EnvPara.FLOAT_PRECISION).view(1, dim, dim)
            if 'virial' not in atoms.info:
                if 'stress' not in atoms.info:
                    try:
                        atoms.info['stress'] = atoms.get_stress()
                    except:
                        pass
                if 'stress' in atoms.info:
                    stress = np.array(atoms.info['stress'])
                    if stress.shape == (6,):
                        stress = np.array([[stress[0], stress[5], stress[4]],
                                        [stress[5], stress[1], stress[3]],
                                        [stress[4], stress[3], stress[2]]])
                    atoms.info['virial'] = -atoms.get_volume() * stress

        padding_shape = {
            'site_energy' : (len(atoms)),
            'energy'      : (1),
            'forces'      : (len(atoms), dim),
            'virial'      : (1, dim, dim),
            'dipole'      : (1, dim),
            'polarizability': (1, dim, dim)
        }
        for key in properties:
            if key in atoms.info:
                data[key + '_t'] = torch.tensor(atoms.info[key], dtype=EnvPara.FLOAT_PRECISION).reshape(padding_shape[key])
                if key + 'weight' in atoms.info:
                    data[key + '_weight'] = torch.tensor(atoms.info[key + '_weight'], dtype=EnvPara.FLOAT_PRECISION).reshape(padding_shape[key])
                else:
                    data[key + '_weight'] = torch.ones(padding_shape[key], dtype=EnvPara.FLOAT_PRECISION)
            else:
                data[key + '_t'] = torch.zeros(padding_shape[key], dtype=EnvPara.FLOAT_PRECISION)
                data[key + '_weight'] = torch.zeros(padding_shape[key], dtype=EnvPara.FLOAT_PRECISION)
        return data

    def __init__(self,
                 indices: Optional[List[int]]=None,
                 cutoff : float=4.0,
                 ) -> None:
        self.indices = indices
        self.cutoff = cutoff

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, idx: int):
        pass

    def subset(self, indices: List[int]):
        ds = copy.copy(self)
        if ds.indices:
            ds.indices = [ds.indices[i] for i in indices]
        else:
            ds.indices = indices
        return ds

def atoms_collate_fn(batch):

    elem = batch[0]
    coll_batch = {}

    for key in elem:
        if key not in ["idx_i", "idx_j"]:
            coll_batch[key] = torch.cat([d[key] for d in batch], dim=0)

    # idx_i and idx_j should to be converted like
    # [0, 0, 1, 1] + [0, 0, 1, 2] -> [0, 0, 1, 1, 2, 2, 3, 4]
    for key in ["idx_i", "idx_j"]:
        coll_batch[key] = torch.cat(
            [batch[i][key] + torch.sum(coll_batch["n_atoms"][:i]) for i in range(len(batch))], dim=0
        )

    coll_batch["batch"] = torch.repeat_interleave(
        torch.arange(len(batch)),
        repeats=coll_batch["n_atoms"].to(torch.long),
        dim=0
    )

    return coll_batch
