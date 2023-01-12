import torch
import torch.nn.functional as F
from typing import Dict, Callable
from .utils import expand_to


class Loss:

    atom_prop = ["forces"]
    structure_prop = ["energy", "virial", "dipole"]

    def __init__(self,
                 weight  : Dict[str, float]={"energy": 1.0, "forces": 1.0},
                 loss_fn : Callable=F.mse_loss,
                 ) -> None:
        self.weight = weight
        self.loss_fn = loss_fn

    def get_loss(self, 
                 batch_data : Dict[str, torch.Tensor],
                 verbose    : bool=False):
        loss = {}
        total_loss = 0.
        for prop in self.weight:
            if prop in self.atom_prop:
                loss[prop] = self.atom_prop_loss(batch_data, prop)
            elif prop in self.structure_prop:
                loss[prop] = self.structure_prop_loss(batch_data, prop)
            total_loss += loss[prop] * self.weight[prop]
        if verbose:
            return total_loss, loss
        return total_loss

    def atom_prop_loss(self,
                       batch_data : Dict[str, torch.Tensor],
                       prop       : str,
                       ) -> torch.Tensor:
        return self.loss_fn(batch_data[f'{prop}_p'], batch_data[f'{prop}_t'])

    def structure_prop_loss(self,
                            batch_data : Dict[str, torch.Tensor],
                            prop       : str,
                            ) -> torch.Tensor:
        n_atoms = expand_to(batch_data['n_atoms'], len(batch_data[f'{prop}_p'].shape))
        return self.loss_fn(batch_data[f'{prop}_p'] / n_atoms, 
                            batch_data[f'{prop}_t'] / n_atoms)


class MissingValueLoss(Loss):

    def atom_prop_loss(self,
                       batch_data : Dict[str, torch.Tensor],
                       prop       : str,
                       ) -> torch.Tensor:
        idx = batch_data[f'has_{prop}'][batch_data['batch']]
        if not torch.any(idx):
            return torch.tensor(0.)
        if torch.all(idx):
            return super().atom_prop_loss(batch_data, prop)
        return self.loss_fn(batch_data[f'{prop}_p'][idx], batch_data[f'{prop}_t'][idx])

    def structure_prop_loss(self,
                            batch_data : Dict[str, torch.Tensor],
                            prop       : str,
                            ) -> torch.Tensor:
        idx = batch_data[f'has_{prop}']
        if not torch.any(idx):
            return torch.tensor(0.)
        if torch.all(idx):
            return super().structure_prop_loss(batch_data, prop)
        n_atoms = expand_to(batch_data['n_atoms'][idx], len(batch_data[f'{prop}_p'].shape))
        return self.loss_fn(batch_data[f'{prop}_p'][idx] / n_atoms, 
                            batch_data[f'{prop}_t'][idx] / n_atoms)
