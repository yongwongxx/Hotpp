import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError
from ase.neighborlist import neighbor_list


class MiaoCalculator(Calculator):

    implemented_properties = [
        "energy",
        "energies",
        "forces",
        "stress",
    ]

    def __init__(self, 
                 cutoff     : float,
                 model_file : str="model.pt",
                 device     : str="cpu",
                 **kwargs,
                 ) -> None:
        Calculator.__init__(self, **kwargs)
        self.cutoff = cutoff
        self.device = device
        self.model = torch.load(model_file).to(device).double()

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        idx_i, idx_j, offsets = neighbor_list("ijS", atoms, self.cutoff, self_interaction=False)
        bonds = np.array([idx_i, idx_j])
        offsets = np.array(offsets) @ atoms.get_cell()
        atomic_number = torch.tensor(atoms.numbers, dtype=torch.long, device=self.device)
        edge_index = torch.tensor(bonds, dtype=torch.long, device=self.device)
        offset = torch.tensor(offsets, dtype=torch.double, device=self.device)
        coordinate = torch.tensor(atoms.positions, dtype=torch.double, device=self.device)
        n_atoms = torch.tensor(len(atoms), dtype=torch.double, device=self.device)
        batch = torch.zeros(len(atoms), dtype=torch.long, device=self.device)
        
        data = {
            "atomic_number" : atomic_number,
            "edge_index"    : edge_index,
            "offset"        : offset,
            "coordinate"    : coordinate,
            "n_atoms"       : n_atoms,
            "batch"         : batch,
        }

        self.model(data, properties, create_graph=False)
        if "energy" in properties:
            self.results["energy"] = data["energy_p"].detach().cpu().numpy()[0]
        if "energies" in properties:
            self.results["energies"] = data["site_energy_p"].detach().cpu().numpy()
        if "forces" in properties:
            self.results["forces"] = data["forces_p"].detach().cpu().numpy()
        if "stress" in properties:
            raise Exception("ni shi gu yi zhao cha shi bu shi?")
            # virial = np.sum(
            #     np.array(self.calc.getVirials()).reshape(9, -1), axis=1)
            # if sum(atoms.get_pbc()) > 0:
            #     stress = -0.5 * (virial.copy() +
            #                      virial.copy().T) / atoms.get_volume()
            #     self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
            # else:
            #     raise PropertyNotImplementedError
