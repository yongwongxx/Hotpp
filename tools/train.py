import logging, time, yaml, os
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from ase.data import atomic_numbers
from hotpp.utils import setup_seed
from hotpp.model import MiaoNet, LitAtomicModule
from hotpp.layer.cutoff import *
from hotpp.layer.embedding import AtomicEmbedding
from hotpp.layer.radial import *
from hotpp.data import LitAtomsDataset

from hotpp.logger import set_logger
set_logger(log_path='log.txt', level='DEBUG')
log = logging.getLogger(__name__)


class LogAllLoss(pl.Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            epoch = trainer.current_epoch

            if epoch == 0:
                content = f"{'epoch':^10}|{'lr':^10}|{'total':^21}"
                for prop in pl_module.p_dict["Train"]['targetProp']:
                    content += f"|{prop:^21}"
                log.info(content)

            if epoch % pl_module.p_dict["Train"]["evalInterval"] == 0:
                lr = trainer.optimizers[0].param_groups[0]["lr"]
                loss_metrics = trainer.callback_metrics
                train_loss = loss_metrics['train_loss']
                val_loss = loss_metrics['val_loss']
                content = f"{epoch:^10}|{lr:^10.2e}|{train_loss:^10.4f}/{val_loss:^10.4f}"
                for prop in pl_module.p_dict["Train"]['targetProp']:
                    prop = "forces" if prop == "direct_forces" else prop
                    content += f"|{loss_metrics[f'train_{prop}']:^10.4f}/{loss_metrics[f'val_{prop}']:^10.4f}"
                log.info(content)

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
    if "direct_forces" in target:
        assert "forces" not in target_way, "Cannot learn forces and direct_forces at the same time"
        target_way["direct_forces"] = 1
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
                    norm_factor=n_neighbor,
                    mode=model_dict['mode'])
    return model


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
            "std": "force",
            "numWorkers": 0,
            "pinMemory": False,
        },
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
        "Train": {
            "learningRate": 0.001,
            "allowMissing": False,
            "targetProp": ["energy", "forces"],
            "weight": [0.1, 1.0],
            "forceScale": 0.,
            "evalInterval": 10,
            "saveInterval": 500,
            "saveStart": 1000,
            "evalTest": True,
            "maxGradNorm": None,
            "Optimizer": {
                "type": "Adam",
                "amsGrad": True,
                "weightDecay": 0.,
                },
            "LrScheduler": {
                "type": "constant",
            },
            "emaDecay": 0.,
        },
    }
    with open(input_file) as f:
        update_dict(p_dict, yaml.load(f, Loader=yaml.FullLoader))

    os.makedirs(p_dict["outputDir"], exist_ok=True)

    with open("allpara.yaml", "w") as f:
        yaml.dump(p_dict, f)

    setup_seed(p_dict["seed"])
    log.info("Using seed {}".format(p_dict["seed"]))

    log.info(f"Preparing data...")
    dataset = LitAtomsDataset(p_dict)
    dataset.setup()

    try:
        mean = dataset.per_energy_mean.detach().cpu().numpy()
    except:
        mean = 0.
    if p_dict["Data"]["std"] == "force": 
        std = dataset.forces_std.detach().cpu().numpy()
    elif p_dict["Data"]["std"] == "energy":
        std = dataset.per_energy_std.detach().cpu().numpy()
    else:
        assert type(std) is float, "std must be 'force', 'energy' or a float!" 
        std = p_dict["Data"]["std"]
    n_neighbor = dataset.n_neighbor_mean.detach().cpu().numpy()
    elements = list(dataset.all_elements.detach().cpu().numpy())
    log.info(f"mean  : {mean}")
    log.info(f"std   : {std}")
    log.info(f"n_neighbor   : {n_neighbor}")
    log.info(f"all_elements : {elements}")
    if load_model is not None:
        model = torch.load(load_model)
    else:
        model = get_model(p_dict, elements, mean, std, n_neighbor)
        model.register_buffer('all_elements', torch.tensor(elements, dtype=torch.long))
        model.register_buffer('cutoff', torch.tensor(p_dict["cutoff"], dtype=torch.float64))

    # log.info(" Network Architecture ".center(100, "="))
    # log.info(model)
    # log.info(f"Number of parameters: {sum([p.numel() for p in model.parameters()])}")
    # log.info("=" * 100)

    lit_model = LitAtomicModule(model=model, p_dict=p_dict)
    from lightning.pytorch.profilers import PyTorchProfiler
    profiler = PyTorchProfiler(
            on_trace_ready=torch.profiler.tensorboard_trace_handler('.'),
            schedule=torch.profiler.schedule(skip_first=10,
                                             wait=1,
                                             warmup=1,
                                             active=20,
                                             repeat=1))

    logger = pl.loggers.TensorBoardLogger(save_dir='.')
    callbacks = [
        ModelCheckpoint(
            dirpath='outDir',
            filename='{epoch}-{val_loss:.2f}',
            every_n_epochs=p_dict["Train"]["evalInterval"],
            save_top_k=1,
            monitor="val_loss"
        ),
        LogAllLoss(),
    ]
    trainer = pl.Trainer(
        profiler=profiler,
        logger=logger,
        callbacks=callbacks,
        default_root_dir='.',
        max_epochs=100,
        enable_progress_bar=False,
        log_every_n_steps=50,
        #num_nodes=1, 
        strategy='ddp_find_unused_parameters_true'
        )

    trainer.fit(lit_model, datamodule=dataset)

if __name__ == "__main__":
    main()
