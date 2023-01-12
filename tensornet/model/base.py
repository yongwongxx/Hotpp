import torch
from torch import nn
from typing import List, Dict
from tensornet.utils import _scatter_add, find_distances, add_scaling


class AtomicModule(nn.Module):

    def __init__(self, 
                 mean  : float=0.,
                 std   : float=1.,
                 ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).float())
        self.register_buffer("std", torch.tensor(std).float())

    def forward(self, 
                batch_data   : Dict[str, torch.Tensor],
                properties   : List[str]=['energy'],
                create_graph : bool=True,
                ) -> Dict[str, torch.Tensor]:
        required_derivatives = []
        if 'forces' in properties:
            required_derivatives.append('coordinate')
            batch_data['coordinate'].requires_grad_()
        if 'virial' in properties or 'stress' in properties:
            required_derivatives.append('scaling')
            batch_data['scaling'].requires_grad_()
            add_scaling(batch_data)
        output_tensors = self.calculate(batch_data)
        if 'site_energy' in output_tensors:
            site_energy = output_tensors['site_energy'] * self.std + self.mean
        if 'dipole' in output_tensors:
            batch_data['dipole_p'] = _scatter_add(output_tensors['dipole'], batch_data['batch'])
        if ('site_energy' in properties) or ('energies' in properties):
            batch_data['site_energy_p'] = site_energy
        if 'energy' in properties:
            batch_data['energy_p'] = _scatter_add(site_energy, batch_data['batch'])
        if len(required_derivatives) > 0:
            grads = torch.autograd.grad(site_energy.sum(),
                                        [batch_data[prop] for prop in required_derivatives],
                                        create_graph=create_graph)
        if 'forces' in properties:
            batch_data['forces_p'] = -grads[required_derivatives.index('coordinate')]
        if 'virial' in properties or 'stress' in properties:
            batch_data['virial_p'] = -grads[required_derivatives.index('scaling')]
        return batch_data

    def calculate(self):
        raise NotImplementedError(f"{self.__class__.__name__} must have 'calculate'!")

