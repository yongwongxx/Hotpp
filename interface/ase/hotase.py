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
        "dipole",
        "polarizability",
    ]

    def __init__(self,
                 model_file : str="model.pt",
                 device     : str="cpu",
                 double     : bool=True,
                 **kwargs,
                 ) -> None:
        Calculator.__init__(self, **kwargs)
        self.device = device
        self.model = torch.jit.load(model_file, map_location=device)
        self.precision = torch.float32
        if double:
            self.model = self.model.double()
            self.precision = torch.double
        self.cutoff = float(self.model.cutoff.detach().cpu().numpy())

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
        offset = np.array(offsets) @ atoms.get_cell()

        data = {
            "atomic_number": torch.tensor(atoms.numbers, dtype=torch.long, device=self.device),
            "idx_i"        : torch.tensor(idx_i, dtype=torch.long, device=self.device),
            "idx_j"        : torch.tensor(idx_j, dtype=torch.long, device=self.device),
            "coordinate"   : torch.tensor(atoms.positions, dtype=self.precision, device=self.device),
            "n_atoms"      : torch.tensor([len(atoms)], dtype=torch.long, device=self.device),
            "offset"       : torch.tensor(offset, dtype=self.precision, device=self.device),
            "scaling"      : torch.eye(3, dtype=self.precision, device=self.device).view(1, 3, 3),
            "batch"        : torch.zeros(len(atoms), dtype=torch.long, device=self.device),
        }

        data = self.model(data, properties, create_graph=False)
        if "energy" in properties:
            self.results["energy"] = data["energy_p"].detach().cpu().numpy()[0]
        if "energies" in properties:
            self.results["energies"] = data["site_energy_p"].detach().cpu().numpy()
        if "forces" in properties:
            self.results["forces"] = data["forces_p"].detach().cpu().numpy()
        if "stress" in properties:
            virial = data["virial_p"].detach().cpu().numpy().reshape(-1)
            if sum(atoms.get_pbc()) > 0:
                stress = -0.5 * (virial.copy() + virial.copy().T) / atoms.get_volume()
                self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
            else:
                raise PropertyNotImplementedError
        if "dipole" in properties:
            self.results["dipole"] = data["dipole_p"].detach().cpu().numpy()
        if "polarizability" in properties:
            self.results["polarizability"] = data["polarizability_p"].detach().cpu().numpy()
