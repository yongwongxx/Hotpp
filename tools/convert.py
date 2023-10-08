import torch
from train import get_model
import yaml
import sys


def convert(ckptfile, device='cpu'):
    ckpt = torch.load(ckptfile, map_location=device)
    state_dict = ckpt['state_dict']
    elements = state_dict['model.all_elements'].detach().cpu().numpy()
    mean = state_dict['model.mean'].detach().cpu().numpy()
    std = state_dict['model.std'].detach().cpu().numpy()
    n_neighbor = state_dict['model.norm_factor'].detach().cpu().numpy()
    cutoff = state_dict['model.cutoff'].detach().cpu().numpy()

    p_dict = {
        "cutoff": cutoff,
        "Model": {
            "mode": "normal",
            "activateFn": "silu",
            "nEmbedding": 64,
            "nLayer": 5,
            "maxRWay": 2,
            "maxOutWay": 2,
            "nHidden": 64,
            "targetWay": {0 : 'site_energy'},
            "CutoffLayer": {
                "type": "poly",
                "p": 5,
            },
            "RadialLayer": {
                "type": "besselMLP",
                "nBasis": 8,
                "nHidden": [64, 64, 64],
                "activateFn": "silu",
            }
        },
    }
    
    with open('input.yaml') as f:
        p_dict['Model'].update(yaml.load(f, Loader=yaml.FullLoader))
    model = get_model(p_dict, elements=elements, mean=mean, std=std, n_neighbor=n_neighbor)
    model.load_state_dict(state_dict)
    torch.save('model.pt', model)
    

if __name__ == "__main__":
    ckptfile = sys.argv[1]
    convert(ckptfile)
