import logging, time, yaml, os
import numpy as np
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from ase.data import atomic_numbers
from ..utils import setup_seed, get_loss
from ..model import MiaoNet
from ..layer import SmoothCosineCutoff, BesselPoly, AtomicEmbedding
from ..data import *


log = logging.getLogger(__name__)


def update_dict(d1, d2):
    for key in d2:
        if key in d1 and isinstance(d1[key], dict):
            update_dict(d1[key], d2[key])
        else:
            d1[key] = d2[key]
    return d1


def get_model(p_dict, elements, mean, std, n_neighbor):
    model_dict = p_dict['Model']
    cut_fn = SmoothCosineCutoff(cutoff=p_dict['cutoff'])
    emb = AtomicEmbedding(elements, model_dict['nEmbedding'])
    radial_fn = BesselPoly(r_max=p_dict['cutoff'], n_max=model_dict['nBasis'])
    model = MiaoNet(embedding_layer=emb,
                    radial_fn=radial_fn,
                    cutoff_fn=cut_fn,
                    n_layers=model_dict['nLayer'],
                    max_r_way=model_dict['maxRWay'],
                    max_out_way=model_dict['maxOutWay'],
                    output_dim=model_dict['nHidden'],
                    activate_fn=model_dict['activateFn'],
                    target_way={0 : 'site_energy'},
                    mean=mean,
                    std=std,
                    norm_factor=n_neighbor).to(p_dict['device'])
    return model


def get_dataset(p_dict):
    data_dict = p_dict['Data']
    if data_dict['type'] == 'rmd17':
        dataset = RevisedMD17(data_dict['path'], 
                              data_dict['name'], 
                              cutoff=p_dict['cutoff'], 
                              device=p_dict['device'])
    if data_dict['type'] == 'ase':
        if 'name' in data_dict:
            dataset = ASEData(root=data_dict['path'],
                              name=data_dict['name'],
                              cutoff=p_dict['cutoff'],
                              device=p_dict['device'])
        else:
            dataset = None
    return dataset


def split_dataset(dataset, p_dict):
    data_dict = p_dict['Data']
    if ("trainSplit" in data_dict) and ("testSplit" in data_dict):
        log.info("Load split from {} and {}".format(data_dict["trainSplit"], data_dict["testSplit"]))
        return dataset.load_split(data_dict["trainSplit"], data_dict["testSplit"])
    if ("trainNum" in data_dict) and (("testNum" in data_dict)):
        log.info("Random split, train num: {}, test num: {}".format(data_dict["trainNum"], data_dict["testNum"]))
        return dataset.random_split(data_dict["trainNum"], data_dict["testNum"])
    if ("trainSet" in data_dict) and ("testSet" in data_dict):
        assert data_dict['type'] == 'ase', "trainset must can be read by ase!"
        trainset = ASEData(root=data_dict['path'], 
                           name=data_dict['trainSet'],
                           cutoff=p_dict['cutoff'],
                           device=p_dict['device'])
        testset = ASEData(root=data_dict['path'], 
                          name=data_dict['testSet'],
                          cutoff=p_dict['cutoff'],
                          device=p_dict['device'])
        return trainset, testset
    raise Exception("No splitting!")


def train_one_step(model, optimizer, batch_data, p_dict):
    optimizer.zero_grad()
    model(batch_data, ['energy', 'forces'])
    loss = get_loss(batch_data, p_dict["Train"]["weight"])
    loss.backward()
    optimizer.step()


def eval(model, data_loader):
    total, e, f = [], [], []
    for i_batch, batch_data in enumerate(data_loader):
        model(batch_data, ['energy', 'forces'], create_graph=False)
        loss, ee, ff = get_loss(batch_data, [1., 1., 0.], verbose=True, loss_fn=F.l1_loss)
        total.append(loss.detach().cpu().numpy())
        e.append(ee.detach().cpu().numpy())
        f.append(ff.detach().cpu().numpy())
    t1 = np.mean(total)
    e1 = np.mean(e)
    f1 = np.mean(f)
    return t1, e1, f1

    
def train(model, optimizer, train_loader, test_loader, p_dict):
    min_loss = 10000
    t = time.time()
    log.info("epoch\ttime\tt_train\tt_test\te_train\te_test\tf_train\tf_test")
    for i in range(p_dict["Train"]["epoch"]):
        for i_batch, batch_data in enumerate(train_loader):
            train_one_step(model, optimizer, batch_data, p_dict)
        if i % p_dict["Train"]["logInterval"] == 0:
            t1, e1, f1 = eval(model, train_loader)
            if p_dict["Train"]["evalTest"]:
                t2, e2, f2 = eval(model, test_loader)
            else:
                t2, e2, f2 = t1, e1, f1
            log.info(f"{i:5}\t{time.time() - t:.2f}\t{t1:.4f}\t{t2:.4f}\t"
                         f"{e1:.4f}\t{e2:.4f}\t{f1:.4f}\t{f2:.4f}")
            if t2 < min_loss:
                min_loss = t2
                torch.save(model, 'model.pt')
                torch.save(model.state_dict(), 'model_state_dict.pt')
                torch.save(optimizer.state_dict(), 'optimizer_state_dict.pt')
            t = time.time()


def main(*args, input_file='input.yaml', restart=False, **kwargs):
    # Default values
    p_dict = {
        "workDir": os.getcwd(),
        "seed": np.random.randint(0, 100000000),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "Data": {
            "path": os.getcwd(),
            "trainBatch": 32,
            "testBatch": 32,
        },
        "Model": {
            "activateFn": "jilu",
            "nEmbedding": 64,
            "nBasis": 20, 
            "nLayer": 5,
            "maxRWay": 2,
            "maxOutWay": 2,
            "nHidden": 64,
        },
        "Train": {
            "lr": 0.001,
            "weight": [0.1, 1.0, 0.0],
            "logInterval": 100,
            "evalTest": True,
        },
    }
    with open(input_file) as f:
        update_dict(p_dict, yaml.load(f, Loader=yaml.FullLoader))

    with open("allpara.yaml", "w") as f:
        yaml.dump(p_dict, f)

    setup_seed(p_dict["seed"])
    log.info("Using seed {}".format(p_dict["seed"]))
    
    dataset = get_dataset(p_dict)
    trainset, testset = split_dataset(dataset, p_dict)
    train_loader = DataLoader(trainset, batch_size=p_dict["Data"]["trainBatch"], shuffle=True)
    test_loader = DataLoader(testset, batch_size=p_dict["Data"]["testBatch"], shuffle=False)
    if dataset is None:
        dataset = trainset

    mean = dataset.per_energy_mean.detach().cpu().numpy()
    std = dataset.forces_std.detach().cpu().numpy()
    n_neighbor = dataset.n_neighbor_mean
    elements = list(dataset.all_elements.detach().cpu().numpy())
    log.info(f"energy_mean  : {mean}")
    log.info(f"forces_std   : {std}")
    log.info(f"n_neighbor   : {n_neighbor}")
    log.info(f"all_elements : {elements}")
    model = get_model(p_dict, elements, mean, std, n_neighbor)
    optimizer = torch.optim.Adam(model.parameters(), lr=p_dict["Train"]["lr"])
    if restart:
        model.load_state_dict(torch.load('model_state_dict.pt'))
        optimizer.load_state_dict(torch.load('optimizer_state_dict.pt'))
    train(model, optimizer, train_loader, test_loader, p_dict)


if __name__ == "__main__":
    main()
