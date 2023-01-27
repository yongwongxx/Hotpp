import torch


def main(*args, model="model.pt", device="cpu", output="infer.pt", **kwargs):
    model = torch.load(model, map_location=torch.device(device))
    infer = torch.jit.script(model)
    for params in infer.parameters():
        params.requires_grad=False
    infer.save(output)
    

if __name__ == "__main__":
    main()
