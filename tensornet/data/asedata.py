from .base import AtomsData
from ..utils import progress_bar
from typing import List, Optional
from ase import Atoms
from ase.io import read
import torch
import os


class ASEData(AtomsData):
    def __init__(self,
                 frames : Optional[List[Atoms]]=None,
                 root   : Optional[str]=None,
                 name   : Optional[str]=None,
                 format : Optional[str]=None,
                 cutoff : float=4.0,
                 device : str="cpu",
                 ) -> None:
        self.cutoff = cutoff
        self.device = device
        if frames is None:
            frames = read(os.path.join(root, name), format=format, index=':')
        self.frames = frames
        self.name = name or "processed"
        if root is None:
            root = os.getcwd()
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=device)

    @property
    def processed_file_names(self) -> str:
        return f"{self.name}.pt"

    def process(self):
        n_data = len(self.frames)
        data_list = []
        for i in range(n_data):
            progress_bar(i, n_data)
            data = self.atoms_to_graph(self.frames[i], self.cutoff, self.device)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, name='{self.name}')"
