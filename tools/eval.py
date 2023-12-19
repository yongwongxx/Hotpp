from ase.io import read                                                                                        
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from hotpp.data import ASEData, ASEDBData, atoms_collate_fn


def eval(model, data_loader, properties, device='cpu'):
    output = {prop: [] for prop in properties}
    target = {prop: [] for prop in properties}
    n_atoms = []
    for batch_data in data_loader:
        for key in batch_data:
            batch_data[key] = batch_data[key].to(device)
        model(batch_data, properties, create_graph=False)
        n_atoms.extend(batch_data['n_atoms'].detach().cpu().numpy())
        for prop in properties:
            output[prop].extend(batch_data[f'{prop}_p'].detach().cpu().numpy())
            if f'{prop}_t' in batch_data:
                target[prop].extend(batch_data[f'{prop}_t'].detach().cpu().numpy())
    for prop in properties:
        np.save(f'output_{prop}.npy', np.array(output[prop]))
        np.save(f'target_{prop}.npy', np.array(target[prop]))
    np.save('n_atoms.npy', np.array(n_atoms))
    return None


def main(*args, cutoff=None, modelfile='model.pt', device='cpu', datafile='data.traj', format=None, 
         properties=["energy", "forces"], batchsize=32, num_workers=4, pin_memory=True, **kwargs):
    model = torch.load(model, map_location=device)
    cutoff = float(model.cutoff.detach().cpu().numpy())
    if '.db' in dataset:
        dataset = ASEDBData(datapath=dataset,
                            properties=properties,
                            cutoff=cutoff)
    else:
        frames = read(dataset, index=':', format=format)
        dataset = ASEData(frames=frames, cutoff=cutoff, properties=properties)
    data_loader = DataLoader(dataset,
                             batch_size=batchsize,
                             shuffle=False,
                             collate_fn=atoms_collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    eval(model, data_loader, properties, device)


if __name__ == "__main__":
    main()
