from .base import AtomsDataset
from typing import List, Optional
from ase import Atoms
from ase.io import read
from ase.db import connect


class ASEData(AtomsDataset):

    def __init__(self,
                 frames     : Optional[List[Atoms]]=None,
                 indices    : Optional[List[int]]=None,
                 properties : Optional[List[str]]=['energy', 'forces'],
                 cutoff     : float=4.0,
                 ) -> None:
        super().__init__(indices=indices, cutoff=cutoff)
        self.frames = frames
        self.properties = properties

    def __len__(self):
        if self.indices is None:
            return len(self.frames)
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        data = self.atoms_to_data(self.frames[idx],
                                  properties=self.properties,
                                  cutoff=self.cutoff)
        return data

    def extend(self, frames):
        self.frames.extend(frames)


class ASEDBData(AtomsDataset):

    def __init__(self,
                 datapath   : Optional[List[Atoms]]=None,
                 indices    : Optional[List[int]]=None,
                 properties : Optional[List[str]]=['energy', 'forces'],
                 cutoff     : float=4.0,
                 ) -> None:
        super().__init__(indices=indices, cutoff=cutoff)
        self.datapath = datapath
        self.conn = connect(self.datapath, use_lock_file=False)
        self.properties = properties

    def __len__(self):
        if self.indices is None:
            return self.conn.count()
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = int(self.indices[idx])
        row = self.conn.get(idx + 1)
        atoms = Atoms(numbers=row['numbers'],
                      cell=row['cell'],
                      positions=row['positions'],
                      pbc=row['pbc'],
                      info=row.data
                      )
        data = self.atoms_to_data(atoms,
                                  properties=self.properties,
                                  cutoff=self.cutoff)
        return data
