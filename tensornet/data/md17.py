import os
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from .base import atoms_to_graph
from ..utils import progress_bar
from ase import Atoms


class RevisedMD17(InMemoryDataset):

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
    def processed_dir(self) -> str:
        return os.path.join(self.root, f"processed_{self.cutoff:.2f}".replace(".", "_"))

    @property
    def raw_file_names(self) -> str:
        return f"rmd17_{self.name}.npz"

    @property
    def processed_file_names(self) -> str:
        return f"rmd17_{self.name}.pt"

    def process(self):
        raw_data = np.load(self.raw_paths[0])
        data_list = []
        n_data = len(raw_data["energies"])
        for i in range(n_data):
            progress_bar(i, n_data)
            atoms = Atoms(symbols=raw_data["nuclear_charges"],
                        positions=raw_data["coords"][i],
                        info={"energy": raw_data["energies"][i],
                              "forces": raw_data["forces"][i]}
                        )
            data = atoms_to_graph(atoms, self.cutoff, self.device)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, name='{self.name}')"
