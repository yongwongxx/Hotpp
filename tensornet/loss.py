import torch
import torch.nn.functional as F
from typing import Dict, Callable
from .utils import expand_to


class Loss:

    atom_prop = ["forces"]
    structure_prop = ["energy", "virial", "dipole", "polarizability"]

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


class ForceScaledLoss(Loss):

    def __init__(self,
                 weight  : Dict[str, float]={"energy": 1.0, "forces": 1.0},
                 loss_fn : Callable=F.mse_loss,
                 scaled  : float=1.0,
                 ) -> None:
        super().__init__(weight, loss_fn)
        self.scaled = scaled

    def atom_prop_loss(self,
                       batch_data : Dict[str, torch.Tensor],
                       prop       : str,
                       ) -> torch.Tensor:
        if 'prop' != 'forces':
            return super().atom_prop_loss(batch_data, prop)
        reweight = self.scaled / (torch.norm(batch_data['force_t'], dim=1) + self.scaled)
        return self.loss_fn(batch_data[f'forces_p'] * reweight, batch_data[f'forces_t'] * reweight)


class MissingValueLoss(Loss):

    def atom_prop_loss(self,
                       batch_data : Dict[str, torch.Tensor],
                       prop       : str,
                       ) -> torch.Tensor:
        # idx = batch_data[f'{prop}_weight'][batch_data['batch']]
        # if not torch.any(idx):
        #     return torch.tensor(0.)
        # if torch.all(idx):
        #     return super().atom_prop_loss(batch_data, prop)
        # return self.loss_fn(batch_data[f'{prop}_p'][idx], batch_data[f'{prop}_t'][idx])
        return self.loss_fn(batch_data[f'{prop}_p'] * batch_data[f'{prop}_weight'],
                            batch_data[f'{prop}_t'] * batch_data[f'{prop}_weight'])

    def structure_prop_loss(self,
                            batch_data : Dict[str, torch.Tensor],
                            prop       : str,
                            ) -> torch.Tensor:
        # idx = batch_data[f'{prop}_weight']
        # if not torch.any(idx):
        #     return torch.tensor(0.)
        # if torch.all(idx):
        #     return super().structure_prop_loss(batch_data, prop)
        # n_atoms = expand_to(batch_data['n_atoms'], len(batch_data[f'{prop}_p'].shape))
        # return self.loss_fn(batch_data[f'{prop}_p'][idx] / n_atoms, 
        #                     batch_data[f'{prop}_t'][idx] / n_atoms)
        weight = batch_data[f'{prop}_weight'] / expand_to(batch_data['n_atoms'], len(batch_data[f'{prop}_p'].shape))
        return self.loss_fn(batch_data[f'{prop}_p'] * weight,
                            batch_data[f'{prop}_t'] * weight)
