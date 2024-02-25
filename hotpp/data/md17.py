import numpy as np
import os
import torch
from .base import AtomsDataset
from typing import List, Optional
from ase import Atoms

__all__ = ["RevisedMD17"]


class RevisedMD17(AtomsDataset):

    file_names = [
        "aspirin",
        "azobenzene",
        "benzene",
        "ethanol",
        "malonaldehyde",
        "naphthalene",
        "paracetamol",
        "salicylic",
        "toluene",
        "uracil",
    ]

    def __init__(self,
                 root: str,
                 name: str,
                 indices    : Optional[List[int]]=None,
                 cutoff: float=4.0
                 ) -> None:
        super().__init__(indices=indices, cutoff=cutoff)
        assert name in self.file_names
        datapath = os.path.join(root, f"rmd17_{name}.npz")
        raw_data = np.load(datapath)
        self.symbols = raw_data["nuclear_charges"]
        self.energy = raw_data["energies"]
        self.forces = raw_data["forces"]
        self.coords = raw_data["coords"]
        self.cutoff = cutoff

    def __len__(self):
        if self.indices is None:
            return len(self.energy)
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = int(self.indices[idx])
        atoms = Atoms(symbols=self.symbols,
                      positions=self.coords[idx],
                      info={"energy": self.energy[idx],
                            "forces": self.forces[idx]}
                      )
        data = self.atoms_to_data(atoms, cutoff=self.cutoff)
        return data
