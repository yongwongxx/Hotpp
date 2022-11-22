from torch.utils.data import Dataset
import torch
import numpy as np
from tensornet.neighbor import get_environment


class AtomsData(Dataset):
    def __init__(self, frames, cutoff):
        self.frames = frames
        self.cutoff = cutoff
        self.datalist = [{} for _ in range(len(frames))]

    def __len__(self):
        return len(self.frames)

    def extend(self, frames):
        self.frames.extend(frames)
        self.datalist.extend([{} for _ in range(len(frames))])

    def __getitem__(self, i):
        if not self.datalist[i]:
            atoms = self.frames[i]
            self.datalist[i] = get_dict(atoms, self.cutoff)
        return self.datalist[i]

    def remove(self, index):
        for i in sorted(index, reverse=True):
            self.frames.pop(i)
            self.datalist.pop(i)


def get_dict(atoms, cutoff):
    neighbor, offset, mask = get_environment(atoms, cutoff)

    d = {
        'neighbor': torch.from_numpy(neighbor).long(),
        'offset': torch.from_numpy(offset).float(),
        'mask': torch.from_numpy(mask).float(),
        'coordinate': torch.tensor(atoms.positions).float(),
        'cell': torch.tensor(atoms.cell[:]).float(),
        'symbol': torch.tensor(atoms.numbers).long(),
        'scaling': torch.eye(3).float(),
    }

    for key in ['energy', 'forces', 'stress']:
        if key in atoms.info:
            d[key] = torch.tensor(atoms.info[key]).float()
    return d


def _collate_atoms(data):
    properties = data[0]
    max_size = {
        prop: np.array(val.size(), dtype=int) for prop, val in properties.items()
    }

    for properties in data[1:]:
        for prop, val in properties.items():
            max_size[prop] = np.maximum(
                max_size[prop], np.array(val.size(), dtype=int)
            )

    batch = {
        p: torch.zeros(len(data), *[int(ss) for ss in size]).type(
            data[0][p].type()
        )
        for p, size in max_size.items()
    }

    for k, properties in enumerate(data):
        for prop, val in properties.items():
            shape = val.size()
            s = (k,) + tuple([slice(0, d) for d in shape])
            batch[prop][s] = val
    return batch


def convert_frames(frames, cutoff):
    return _collate_atoms([get_dict(atoms, cutoff) for atoms in frames])
