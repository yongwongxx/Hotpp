from torch.utils.data import Dataset
import torch
import numpy as np
from tensornet.neighbor import get_environment
from tensornet.utils import find_distances, EnvPara
import os


class AtomsData(Dataset):
    def __init__(self, frames, cutoff, device='cpu'):
        self.frames = frames
        self.cutoff = cutoff
        self.datalist = [{} for _ in range(len(frames))]
        self.device = device

    def __len__(self):
        return len(self.frames)

    def extend(self, frames):
        self.frames.extend(frames)
        self.datalist.extend([{} for _ in range(len(frames))])

    def __getitem__(self, i):
        if not self.datalist[i]:
            atoms = self.frames[i]
            self.datalist[i] = get_dict(atoms, self.cutoff, self.device)
        return self.datalist[i]

    def remove(self, index):
        for i in sorted(index, reverse=True):
            self.frames.pop(i)
            self.datalist.pop(i)


def get_dict(atoms, cutoff, device='cpu'):
    neighbor, offset, mask = get_environment(atoms, cutoff)

    d = {
        'neighbor': torch.tensor(neighbor, dtype=torch.long, device=device),
        'offset': torch.tensor(offset, dtype=EnvPara.FLOAT_PRECISION, device=device),
        'mask': torch.tensor(mask, dtype=torch.bool, device=device),
        'coordinate': torch.tensor(atoms.positions, dtype=EnvPara.FLOAT_PRECISION, device=device),
        'cell': torch.tensor(atoms.cell[:], dtype=EnvPara.FLOAT_PRECISION, device=device),
        'atomic_number': torch.tensor(atoms.numbers, dtype=torch.long, device=device),
        'scaling': torch.eye(3, dtype=EnvPara.FLOAT_PRECISION, device=device),
        'n_atoms': torch.tensor(len(atoms), dtype=EnvPara.FLOAT_PRECISION, device=device)
    }

    for key in ['energy', 'forces']:#, 'stress']:
        if key in atoms.info:
            d[key + '_t'] = torch.tensor(atoms.info[key], dtype=EnvPara.FLOAT_PRECISION, device=device)
    return d


def _collate_atoms(data):
    example = data[0]
    max_size = {
        prop: np.array(val.size(), dtype=int) for prop, val in example.items()
    }

    for d in data[1:]:
        for prop, val in d.items():
            max_size[prop] = np.maximum(
                max_size[prop], np.array(val.size(), dtype=int)
            )

    batch = {
        p: torch.zeros(size=(len(data), *[int(ss) for ss in size]), 
                       dtype=example[p].dtype,
                       device=example[p].device)
        for p, size in max_size.items()
    }

    for k, properties in enumerate(data):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val
    
    batch['mask'] = ~batch['mask']   # True means no atoms 
    return batch


def convert_frames(frames, cutoff, device='cpu'):
    return _collate_atoms([get_dict(atoms, cutoff, device) for atoms in frames])
