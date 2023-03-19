from ase.io import read                                                                                        
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tensornet.data import ASEData, PtData


def eval(model, data_loader, properties):
    output = {prop: [] for prop in properties}
    target = {prop: [] for prop in properties}
    n_atoms = []
    for batch_data in data_loader:
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


def main(*args, cutoff=None, model='model.pt', device='cpu', dataset='data.traj', format=None, 
         properties=["energy", "forces"], batchsize=32, **kwargs):
    if '.pt' in dataset:
        dataset = PtData(dataset, device=device)
    else:
        assert cutoff is not None, "Must have cutoff!!"
        frames = read(dataset, index=':', format=format)
        dataset = ASEData(frames, name="eval_process", cutoff=cutoff, device=device)
    data_loader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
    model = torch.load(model) 
    eval(model, data_loader, properties)


if __name__ == "__main__":
    main()
