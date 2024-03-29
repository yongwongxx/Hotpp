from ase.neighborlist import neighbor_list
import torch
import numpy as np
import spglib
import ase
from ase import Atoms
from ase.cell import Cell
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections


def read_force_constants(file):
    with open(file) as f:
        n = int(f.readline().split()[0])
        force_constants = np.zeros((n, n, 3, 3))
        for i in range(n * n):
            x, y = f.readline().split()
            x = int(x) - 1
            y = int(y) - 1
            force_constants[x, y, 0] = list(map(float, f.readline().split()[:3]))
            force_constants[x, y, 1] = list(map(float, f.readline().split()[:3]))
            force_constants[x, y, 2] = list(map(float, f.readline().split()[:3]))
    return force_constants
            
            
def plot_band_structure(band_dict, axs, color='g', linestyle='-', max_freq=50):
    labels_path = band_dict['labels_path']
    frequencies = band_dict['frequencies']
    distances = band_dict['distances']

    max_dist = distances[-1][-1]
    xscale = max_freq / max_dist * 1.5
    distances_scaled = [d * xscale for d in distances]

    n = 0
    axs[0].set_ylabel("Frequency", fontsize=14)
    for i, path in enumerate(labels_path):
        axs[i].spines['bottom'].set_linewidth(1.5)
        axs[i].spines['left'].set_linewidth(1.5)
        axs[i].spines['right'].set_linewidth(1.5)
        axs[i].spines['top'].set_linewidth(1.5)
        axs[i].tick_params(labelsize=14)
        xticks = [distances_scaled[n][0]]
        for label in path[:-1]:
            xticks.append(distances_scaled[n][-1])
            axs[i].plot([distances_scaled[n][-1], distances_scaled[n][-1]], 
                        [0, max_freq],
                        linewidth=2,
                        linestyle=":",
                        c='grey')
            axs[i].plot(distances_scaled[n], 
                        frequencies[n], 
                        linewidth=2,
                        linestyle=linestyle,
                        c=color)
            n += 1
        axs[i].set_xlim(xticks[0], xticks[-1])
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels(path)
        axs[i].plot([xticks[0], xticks[-1]], 
                    [0, 0], 
                    linewidth=1,
                    c='black')
    return axs

def ase2phono(atoms):
    return PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                        cell=atoms.cell.array,
                        scaled_positions=atoms.get_scaled_positions())

def phono2ase(cell):
    return Atoms(symbols=cell.get_chemical_symbols(),
                 scaled_positions=cell.get_scaled_positions(),
                 cell=cell.get_cell(),
                 pbc=True)


def get_force_constants(calc, phonon):
    phonon.generate_displacements(distance=0.01)
    supercells = phonon.get_supercells_with_displacements()
    set_of_forces = []
    for cell in supercells:
        forces = calc.get_forces(phono2ase(cell))
        forces -= np.mean(forces, axis=0)
        set_of_forces.append(forces)
    set_of_forces = np.array(set_of_forces)
    phonon.produce_force_constants(forces=set_of_forces)
    return phonon.force_constants

def get_band_structure(phonon, atoms):
    cell = Cell(spglib.find_primitive(atoms)[0])
    special_points = cell.get_bravais_lattice().get_special_points()
    labels_path = ase.dft.kpoints.parse_path_string(cell.bandpath().path)
    labels, path = [], []
    for label_path in labels_path:
        p = []
        for l in label_path:
            labels.append(l)
            p.append(special_points[l].tolist())
        path.append(p)
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
    phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
    bands_dict = phonon.get_band_structure_dict()
    bands_dict['labels_path'] = labels_path
    return bands_dict


def get_direct_force_constants(model, atoms):
    device = next(model.parameters()).device
    cutoff = float(model.cutoff.detach().cpu().numpy())
    idx_i, idx_j, offsets = neighbor_list("ijS", atoms, cutoff, self_interaction=False)
    offsets = np.array(offsets) @ atoms.get_cell()
    atomic_number = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
    idx_i = torch.tensor(idx_i, dtype=torch.long, device=device)
    idx_j = torch.tensor(idx_j, dtype=torch.long, device=device)
    offset = torch.tensor(offsets, dtype=torch.double, device=device)
    coordinate = torch.tensor(atoms.positions, dtype=torch.double, device=device)
    n_atoms = torch.tensor(len(atoms), dtype=torch.double, device=device)
    batch = torch.zeros(len(atoms), dtype=torch.long, device=device)

    def get_energy(coordinate):
        batch_data = {"atomic_number" : atomic_number,
                      "idx_i"    : idx_i,
                      "idx_j"     :idx_j,
                    "offset"        : offset,
                    "coordinate"    : coordinate,
                    "n_atoms"       : n_atoms,
                    "batch"         : batch,
                    }
        output_tensors = model.calculate(batch_data)
        return output_tensors['site_energy'].sum()
        # return (output_tensors['site_energy'] * model.std + model.mean).sum()

    hessian = torch.autograd.functional.hessian(get_energy, coordinate).permute(0, 2, 1, 3)
    hessian = hessian.detach().cpu().numpy()
    return hessian
