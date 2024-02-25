import logging
import os
import numpy as np
from copy import copy
from typing import Optional, List, Dict, Tuple, Union
from . import *
from ase.io import read
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


log = logging.getLogger(__name__)


class LitAtomsDataset(pl.LightningDataModule):

    def __init__(self, p_dict):
        super().__init__()
        self.p_dict = p_dict
        self._train_dataloader = None
        self._test_dataloader = None
        self.stats = {}

    def setup(self, stage: Optional[str] = None):
        self.dataset = self.get_dataset()
        self.trainset, self.testset = self.split_dataset()
        # self.calculate_stats()

    def get_dataset(self):
        data_dict = self.p_dict['Data']
        if data_dict['type'] == 'rmd17':
            dataset = RevisedMD17(data_dict['path'], 
                                  data_dict['name'], 
                                  cutoff=self.p_dict['cutoff'])
        if data_dict['type'] == 'ase':
            if 'name' in data_dict:
                frames = read(os.path.join(data_dict['path'], data_dict['name']), index=':')
            else:
                frames = []
            dataset = ASEData(frames=frames,
                              properties=self.p_dict['Train']['targetProp'],
                              cutoff=self.p_dict['cutoff'])
        elif data_dict['type'] == 'ase-db':
            dataset = ASEDBData(datapath=os.path.join(data_dict['path'], data_dict['name']),
                                properties=self.p_dict['Train']['targetProp'],
                                cutoff=self.p_dict['cutoff'])
        return dataset

    def split_dataset(self):
        data_dict = self.p_dict['Data']
        if ("trainSplit" in data_dict) and ("testSplit" in data_dict):
            log.info("Load split from {} and {}".format(data_dict["trainSplit"], data_dict["testSplit"]))
            train_idx = np.loadtxt(data_dict["trainSplit"], dtype=int)
            test_idx  = np.loadtxt(data_dict["testSplit"], dtype=int)
            return self.dataset.subset(train_idx), self.dataset.subset(test_idx)

        if ("trainNum" in data_dict) and (("testNum" in data_dict)):
            log.info("Random split, train num: {}, test num: {}".format(data_dict["trainNum"], data_dict["testNum"]))
            assert data_dict['trainNum'] + data_dict['testNum'] <= len(self.dataset)
            idx = np.random.choice(len(self.dataset), data_dict['trainNum'] + data_dict['testNum'], replace=False)
            train_idx = idx[:data_dict['trainNum']]
            test_idx  = idx[data_dict['trainNum']:]
            return self.dataset.subset(train_idx), self.dataset.subset(test_idx)

        if ("trainSet" in data_dict) and ("testSet" in data_dict):
            assert data_dict['type'] == 'ase', "trainSet and testSet must can be read by ase!"
            trainset = read(os.path.join(data_dict['path'], data_dict['trainSet']), index=':')
            self.dataset.extend(trainset)
            testset = read(os.path.join(data_dict['path'], data_dict['testSet']), index=':')
            self.dataset.extend(testset)
            train_idx = [i for i in range(len(trainset))]
            test_idx = [i for i in range(len(trainset), len(trainset) + len(testset))]
            return self.dataset.subset(train_idx), self.dataset.subset(test_idx)

        raise Exception("No splitting!")

    def train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = DataLoader(self.trainset,
                          batch_size=self.p_dict["Data"]["trainBatch"],
                          shuffle=True,
                          collate_fn=atoms_collate_fn,
                          num_workers=self.p_dict["Data"]["numWorkers"],
                          pin_memory=self.p_dict["Data"]["pinMemory"])
            log.debug(f'numWorkers: {self.p_dict["Data"]["numWorkers"]}')
        return self._train_dataloader

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        if self._test_dataloader is None:
            self._test_dataloader = DataLoader(self.testset,
                          batch_size=self.p_dict["Data"]["testBatch"],
                          shuffle=False,
                          collate_fn=atoms_collate_fn,
                          num_workers=self.p_dict["Data"]["numWorkers"],
                          pin_memory=self.p_dict["Data"]["pinMemory"])
        return self._test_dataloader

    def calculate_stats(self):
        # To be noticed, we assume that the average force is always 0, 
        # so the final result may differ from the actual variance

        N_batch = 0
        N_forces = 0
        per_energy_mean = 0.
        n_neighbor_mean = 0.
        per_energy_std = 0.
        forces_std = 0.
        all_elements = torch.tensor([], dtype=torch.long)#, device=self.p_dict['device'])

        for batch_data in self.train_dataloader():
            # all elemetns
            all_elements = torch.unique(torch.cat((all_elements, batch_data['atomic_number'])))

            # per_energy_mean
            batch_size = batch_data["energy_t"].numel()
            pe = batch_data["energy_t"] / batch_data["n_atoms"]
            pe_mean = torch.mean(pe)
            delta_pe_mean = pe_mean - per_energy_mean
            per_energy_mean += delta_pe_mean * batch_size / (N_batch + batch_size)

            # per_energy_std
            pe_m2 = torch.sum((pe - pe_mean) ** 2)
            pe_corr = batch_size * N_batch / (N_batch + batch_size)
            per_energy_std += pe_m2 + delta_pe_mean ** 2 * pe_corr

            # n_neighbor_mean
            nn_mean = batch_data["idx_i"].shape[0] / torch.sum(batch_data["n_atoms"])
            delta_nn_mean = nn_mean - n_neighbor_mean
            n_neighbor_mean += delta_nn_mean * batch_size / (N_batch + batch_size)
            N_batch += batch_size

            # forces_std
            if 'forces_t' in batch_data:
                forces_size = batch_data["forces_t"].numel()
                forces_m2 = torch.sum(batch_data["forces_t"] ** 2)
                forces_std += forces_m2
                N_forces += forces_size

        per_energy_std = torch.sqrt(per_energy_std / N_batch)
        if N_forces > 0:
            forces_std = torch.sqrt(forces_std / N_forces)

        self.stats["per_energy_mean"] = per_energy_mean
        self.stats["per_energy_std"] = per_energy_std
        self.stats["n_neighbor_mean"] = n_neighbor_mean
        self.stats["forces_std"] = forces_std
        self.stats["all_elements"] = all_elements

    @property
    def per_energy_mean(self):
        if "per_energy_mean" not in self.stats:
            self.calculate_stats()
        return self.stats["per_energy_mean"]

    @property
    def per_energy_std(self):
        if "per_energy_std" not in self.stats:
            self.calculate_stats()
        return self.stats["per_energy_std"]

    @property
    def forces_std(self):
        if "forces_std" not in self.stats:
            self.calculate_stats()
        return self.stats["forces_std"]

    @property
    def n_neighbor_mean(self):
        if "forces_std" not in self.stats:
            self.calculate_stats()
        return self.stats["n_neighbor_mean"]

    @property
    def all_elements(self):
        if "all_elements" not in self.stats:
            self.calculate_stats()
        return self.stats["all_elements"]
