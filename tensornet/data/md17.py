import os
import numpy as np
import torch
from .base import AtomsData
from ..utils import progress_bar
from ase import Atoms

__all__ = ["RevisedMD17"]


class RevisedMD17(AtomsData):

    file_names = [
        "aspirin",
        "azobenzene",
        "benzene",
        "ethanol",
        "malonaldehyd",
        "naphthalene",
        "paracetamol",
        "salicyli",
        "toluene",
        "uracil",
    ]

    def __init__(self, root: str, name: str, cutoff: float=4.0, device: str="cpu") -> None:
        self.name = name
        self.cutoff = cutoff
        self.device = device
        assert name in self.file_names
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=device)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "npz_data")

    @property
    def raw_file_names(self) -> str:
        return f"rmd17_{self.name}.npz"

    @property
    def processed_file_names(self) -> str:
        return f"rmd17_{self.name}.pt"

    def process(self):
        raw_data = np.load(self.raw_paths[0])
        symbols = raw_data["nuclear_charges"]
        coords = raw_data["coords"]
        energies = raw_data["energies"]
        forces = raw_data["forces"]
        data_list = []
        n_data = len(energies)
        for i in range(n_data):
            progress_bar(i, n_data)
            atoms = Atoms(symbols=symbols,
                          positions=coords[i],
                          info={"energy": energies[i],
                                "forces": forces[i]}
                          )
            data = self.atoms_to_graph(atoms, self.cutoff, self.device)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, name='{self.name}')"
