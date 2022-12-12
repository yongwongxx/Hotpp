import torch
from scipy.io import loadmat
from .base import AtomsData
from ..utils import progress_bar
from ase import Atoms


class QM7b(AtomsData):
    def __init__(self, root:str, cutoff:float=4.0, device:str="cpu") -> None:
        self.cutoff = cutoff
        self.device = device 
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=device)

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        return self.root
    
    @property
    def raw_file_names(self) -> str:
        return 'qm7b.mat'

    @property
    def processed_file_names(self) -> str:
        return f'data_{self.cutoff:.2f}'.replace(".", "") + '.pt'

    def process(self):
        raw_data = loadmat(self.raw_paths[0])
        data_list = []
        n_data = len(raw_data["T"][0])
        for i in range(n_data):
            progress_bar(i, n_data)
            n_atoms = sum(raw_data['Z'][i] > 0)
            atoms = Atoms(symbols=raw_data["Z"][i][:n_atoms],
                          positions=raw_data["R"][i][:n_atoms],
                          info={"energy": raw_data["T"][0][i]}
                          )
            data = self.atoms_to_graph(atoms, self.cutoff, self.device)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])
