import torch
from ase.data import chemical_symbols, atomic_numbers


def main(*args, model="model.pt", device="cpu", output="infer.pt", symbols=None, **kwargs):
    model = torch.load(model, map_location=torch.device(device))
    # change embedding layer
    if symbols is not None:
        all_elements = [atomic_numbers[s] for s in symbols]
    else:
        all_elements = model.all_elements.cpu().numpy()
    new_weight = torch.zeros(len(all_elements), model.embedding_layer.n_channel)
    for i, n in enumerate(all_elements):
        new_weight[i] = model.embedding_layer.z_weights.weight.data[n]
    model.embedding_layer.z_weights.weight.data = new_weight
    infer = torch.jit.script(model)
    for params in infer.parameters():
        params.requires_grad=False
    infer.save(output)
    

if __name__ == "__main__":
    main()
