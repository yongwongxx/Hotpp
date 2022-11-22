from ase.neighborlist import NeighborList, NewPrimitiveNeighborList
import numpy as np

def get_environment(atoms, cutoff):
    n_atoms = len(atoms)
    nb = NeighborList([0.5 * cutoff] * n_atoms,
                      skin=0.0, self_interaction=False, bothways=True,
                      primitive=NewPrimitiveNeighborList)
    nb.update(atoms)
    neighborhood_idx, offsets, nbh = [], [], []
    for i in range(n_atoms):
        idx, offset = nb.get_neighbors(i)
        nbh.append(len(idx))
        neighborhood_idx.append(list(idx))
        offsets.append(offset.tolist())
    mask = [None] * n_atoms
    max_nbh = max(max(nbh), 1)
    for i in range(n_atoms):
        neighborhood_idx[i].extend([-1.] * (max_nbh - nbh[i]))
        offsets[i].extend([[0., 0., 0.]] * (max_nbh - nbh[i]))
        mask[i] = [True] * nbh[i] + [False] * (max_nbh - nbh[i])
    neighborhood_idx = np.array(neighborhood_idx, dtype=np.float32)
    offsets = np.array(offsets, dtype=np.float32)
    mask = np.array(mask, dtype=bool)
    return neighborhood_idx, offsets, mask
