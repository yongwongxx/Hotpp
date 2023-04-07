import logging, time, yaml, os
import numpy as np
from torch_geometric.loader import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from ase.data import atomic_numbers
from ..utils import setup_seed
from ..loss import Loss, MissingValueLoss
from ..model import MiaoNet
from ..layer.cutoff import *
from ..layer.embedding import AtomicEmbedding
from ..layer.radial import *
from ..data import *


log = logging.getLogger(__name__)


def update_dict(d1, d2):
    for key in d2:
        if key in d1 and isinstance(d1[key], dict):
            update_dict(d1[key], d2[key])
        else:
            d1[key] = d2[key]
    return d1


def get_cutoff(p_dict):
    cutoff = p_dict['cutoff']
    cut_dict = p_dict['Model']['CutoffLayer']
    if cut_dict['type'] == "cos":
        return CosineCutoff(cutoff=cutoff)
    elif cut_dict['type'] == "cos2":
        return SmoothCosineCutoff(cutoff=cutoff, cutoff_smooth=cut_dict['smoothCutoff'])
    elif cut_dict['type'] == "poly":
        return PolynomialCutoff(cutoff=cutoff, p=cut_dict['p'])
    else:
        raise Exception("Unsupported cutoff type: {}, please choose from cos, cos2, and poly!".format(cut_dict['type']))
    

def get_radial(p_dict, cutoff_fn):
    cutoff = p_dict['cutoff']
    radial_dict = p_dict['Model']['RadialLayer']
    if "bessel" in radial_dict['type']:
        radial_fn = BesselPoly(r_max=cutoff, n_max=radial_dict['nBasis'], cutoff_fn=cutoff_fn)
    elif "chebyshev" in radial_dict['type']:
        if "minDist" in radial_dict:
            r_min = radial_dict['minDist']
        else:
            r_min = 0.5
            log.warning("You are using chebyshev poly as basis function, but does not given 'minDist', "
                        "this may cause some problems!")
        radial_fn = ChebyshevPoly(r_max=cutoff, r_min=r_min, n_max=radial_dict['nBasis'], cutoff_fn=cutoff_fn)
    else:
        raise Exception("Unsupported radial type: {}!".format(radial_dict['type']))
    if "MLP" in radial_dict['type']:
        if radial_dict["activateFn"] == "silu":
            activate_fn = nn.SiLU()
        elif radial_dict["activateFn"] == "relu":
            activate_fn = nn.ReLU()
        else:
            raise Exception("Unsupported activate function in radial type: {}!".format(radial_dict["activateFn"]))
        return MLPPoly(n_hidden=radial_dict['nHidden'], radial_fn=radial_fn, activate_fn=activate_fn)
    else:
        return radial_fn


def get_model(p_dict, elements, mean, std, n_neighbor):
    model_dict = p_dict['Model']
    target = p_dict['Train']['targetProp']
    target_way = {}
    if ("energy" in target) or ("forces" in target) or ("virial" in target):
        target_way["site_energy"] = 0
    if "dipole" in target:
        target_way["dipole"] = 1
    cut_fn = get_cutoff(p_dict)
    emb = AtomicEmbedding(elements, model_dict['nEmbedding'])  # only support atomic embedding now
    radial_fn = get_radial(p_dict, cut_fn)
    model = MiaoNet(embedding_layer=emb,
                    radial_fn=radial_fn,
                    n_layers=model_dict['nLayer'],
                    max_r_way=model_dict['maxRWay'],
                    max_out_way=model_dict['maxOutWay'],
                    output_dim=model_dict['nHidden'],
                    activate_fn=model_dict['activateFn'],
                    target_way=target_way,
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


def get_loss_calculator(p_dict):
    train_dict = p_dict['Train']
    target = train_dict['targetProp']
    weight = train_dict['weight']
    weights = {p: w for p, w in zip(target, weight)}
    if train_dict['allowMissing']:
        return MissingValueLoss(weights, loss_fn=F.mse_loss)
    else:
        return Loss(weights, loss_fn=F.mse_loss)


def eval(model, properties, loss_calculator, data_loader):
    total = []
    prop_loss = {}
    for i_batch, batch_data in enumerate(data_loader):
        model(batch_data, properties, create_graph=False)
        loss, loss_dict = loss_calculator.get_loss(batch_data, verbose=True)
        total.append(loss.detach().cpu().numpy())
        for prop in loss_dict:
            if prop not in prop_loss:
                prop_loss[prop] = []
            prop_loss[prop].append(loss_dict[prop].detach().cpu().numpy())
    t1 = np.sqrt(np.mean(total))
    for prop in loss_dict:
        prop_loss[prop] = np.sqrt(np.mean(prop_loss[prop]))
    return t1, prop_loss

    
def train(model, loss_calculator, optimizer, lr_scheduler, ema, train_loader, test_loader, p_dict):
    min_loss = 10000
    t2 = 10000
    t = time.time()
    content = f"{'epoch':^10}|{'time':^10}|{'lr':^10}|{'total':^21}"
    for prop in p_dict["Train"]['targetProp']:
        content += f"|{prop:^21}"
    log.info(content)
    for epoch in range(p_dict["Train"]["epoch"]):
        for i_batch, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            model(batch_data, p_dict["Train"]['targetProp'])
            loss = loss_calculator.get_loss(batch_data)
            loss.backward()
            if p_dict["Train"]["maxGradNorm"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=p_dict["Train"]["maxGradNorm"])
            optimizer.step()
            if ema is not None:
                ema.update_parameters(model)
        lr_scheduler.step(epoch=epoch, metrics=t2)
        if epoch % p_dict["Train"]["logInterval"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            t1, prop_loss1 = eval(model, p_dict["Train"]['targetProp'], loss_calculator, train_loader)
            if p_dict["Train"]["evalTest"]:
                if ema:
                    t2, prop_loss2 = eval(ema, p_dict["Train"]['targetProp'], loss_calculator, test_loader)
                else:
                    t2, prop_loss2 = eval(model, p_dict["Train"]['targetProp'], loss_calculator, test_loader)
            else:
                t2, prop_loss2 = t1, prop_loss1
            content = f"{epoch:^10}|{time.time() - t:^10.2f}|{lr:^10.2e}|{t1:^10.4f}/{t2:^10.4f}"
            t = time.time()
            for prop in p_dict["Train"]['targetProp']:
                content += f"|{prop_loss1[prop]:^10.4f}/{prop_loss2[prop]:^10.4f}"
            log.info(content)
            if t2 < min_loss:
                min_loss = t2
                save_checkpoints(p_dict["outputDir"], "best", model, ema, optimizer, lr_scheduler)
        if epoch > p_dict["Train"]["saveStart"] and epoch % p_dict["Train"]["saveInterval"] == 0:
            save_checkpoints(p_dict["outputDir"], epoch, model, ema, optimizer, lr_scheduler)


def save_checkpoints(path, name, model, ema, optimizer, lr_scheduler):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    torch.save(checkpoint, os.path.join(path, f"state_dict-{name}.pt"))
    torch.save(model, os.path.join(path, f"model-{name}.pt"))


def get_optimizer(p_dict, model):
    opt_dict = p_dict["Train"]["Optimizer"]
    decay_interactions = {}
    no_decay_interactions = {}
    for name, param in model.son_equivalent_layers.named_parameters():
        if "weight" in name:
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    param_options = dict(
        params=[
            {
                "name": "embedding",
                "params": model.embedding_layer.parameters(),
                "weight_decay": 0.0,
            },
            {
                "name": "interactions_decay",
                "params": list(decay_interactions.values()),
                "weight_decay": opt_dict["weightDecay"],
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "readouts",
                "params": model.readout_layer.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=opt_dict["learningRate"],
        amsgrad=opt_dict["amsGrad"],
    )

    if opt_dict['type'] == "Adam":
        return torch.optim.Adam(**param_options)
    elif opt_dict['type'] == "AdamW":
        return torch.optim.AdamW(**param_options)
    else:
        raise Exception("Unsupported optimizer: {}!".format(opt_dict["type"]))

def get_lr_scheduler(p_dict, optimizer):
    class LrScheduler:
        def __init__(self, p_dict, optimizer) -> None:
            lr_dict = p_dict["Train"]["LrScheduler"]
            self.mode = lr_dict['type']
            if lr_dict['type'] == "exponential":
                self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, 
                                                                           gamma=lr_dict['gamma'])
            elif lr_dict['type'] == "reduceOnPlateau":
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                               factor=lr_dict['lrFactor'],
                                                                               patience=lr_dict['patience'])
            elif lr_dict['type'] != "constant":
                raise Exception("Unsupported LrScheduler: {}!".format(lr_dict['type']))

        def step(self, metrics=None, epoch=None):
            if self.mode == "exponential":
                self.lr_scheduler.step(epoch=epoch)
            elif self.mode == "reduceOnPlateau":
                self.lr_scheduler.step(metrics=metrics, epoch=epoch)

        def state_dict(self):
            return {key: value for key, value in self.lr_scheduler.__dict__.items() if key != 'optimizer'}

        def load_state_dict(self, state_dict):
            self.lr_scheduler.__dict__.update(state_dict)

        def __repr__(self):
            return self.lr_scheduler.__repr__()

    return LrScheduler(p_dict, optimizer)


def main(*args, input_file='input.yaml', load_model=None, load_checkpoint=None, **kwargs):
    # Default values
    p_dict = {
        "workDir": os.getcwd(),
        "seed": np.random.randint(0, 100000000),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "outputDir": os.path.join(os.getcwd(), "outDir"),
        "Data": {
            "path": os.getcwd(),
            "trainBatch": 32,
            "testBatch": 32,
        },
        "Model": {
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
        "Train": {
            "allowMissing": False,
            "targetProp": ["energy", "forces"],
            "weight": [0.1, 1.0, 0.0],
            "logInterval": 100,
            "saveInterval": 500,
            "saveStart": 1000,
            "evalTest": True,
            "maxGradNorm": 10.,
            "Optimizer": {
                "type": "Adam",
                "amsGrad": True,
                "weightDecay": 0.,
                "learningRate": 0.001,
                },
            "LrScheduler": {
                "type": "constant",
            "emaDecay": 0., 
            }
        },
    }
    with open(input_file) as f:
        update_dict(p_dict, yaml.load(f, Loader=yaml.FullLoader))

    os.makedirs(p_dict["outputDir"], exist_ok=True)

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

    try:
        mean = dataset.per_energy_mean.detach().cpu().numpy()
    except:
        mean = 0.
    try:
        std = dataset.forces_std.detach().cpu().numpy()
    except:
        std = 1.
    n_neighbor = dataset.n_neighbor_mean
    elements = list(dataset.all_elements.detach().cpu().numpy())
    log.info(f"energy_mean  : {mean}")
    log.info(f"forces_std   : {std}")
    log.info(f"n_neighbor   : {n_neighbor}")
    log.info(f"all_elements : {elements}")
    if load_model is not None:
        model = torch.load(load_model)
    else:
        model = get_model(p_dict, elements, mean, std, n_neighbor)
        model.register_buffer('all_elements', torch.tensor(elements, dtype=torch.long))
        model.register_buffer('cutoff', torch.tensor(p_dict["cutoff"], dtype=torch.float64))

    optimizer = get_optimizer(p_dict, model)
    lr_scheduler = get_lr_scheduler(p_dict, optimizer)
    if load_checkpoint is not None:
        state_dict = torch.load(load_checkpoint)
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

    log.info(" Network Architecture ".center(100, "="))
    log.info(model)
    log.info(f"Number of parameters: {sum([p.numel() for p in model.parameters()])}")
    log.info(" Optimizer ".center(100, "="))
    log.info(optimizer)
    log.info(" LRScheduler ".center(100, "="))
    log.info(lr_scheduler)

    ema_decay = p_dict["Train"]["emaDecay"]
    if ema_decay > 0:
        ema_avg = lambda averaged_para, para, n: ema_decay * averaged_para + (1 - ema_decay) * para
        ema = AveragedModel(model=model, device=p_dict["device"], avg_fn=ema_avg, use_buffers=False)
        # log.info(" ExponentialMovingAverage ".center(80, "="))
        # log.info(ema)
    else:
        ema = None
    log.info("=" * 100)
    loss_calculator = get_loss_calculator(p_dict)
    train(model, loss_calculator, optimizer, lr_scheduler, ema, train_loader, test_loader, p_dict)


if __name__ == "__main__":
    main()
