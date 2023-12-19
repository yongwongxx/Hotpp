from scipy.io import loadmat
from scipy.stats import special_ortho_group
from ase import Atoms
from hotpp.layer import *
from hotpp.model import MiaoNet
from hotpp.utils import setup_seed, multi_outer_product, EnvPara
from hotpp.data import AtomsData
from torch_geometric.data import Batch
import unittest
import numpy as np
import torch


class TestMolecule(unittest.TestCase):
    def setUp(self):
        setup_seed(0)
        self.max_in_way = 2        # 0: scalar  1: vector  2: matrix  ...
        self.r_min = 1.5
        self.cutoff = 3.5

        dim = 3
        self.R = special_ortho_group.rvs(dim=dim)

        datapath = '../dataset/qm7/qm7b.mat'
        raw_data = loadmat(datapath)
        coordinate = raw_data['R'][0][:6]
        symbol = raw_data['Z'][0][:6]
        atoms1 = Atoms(positions=coordinate, symbols=symbol)
        atoms2 = Atoms(positions=coordinate @ self.R.T, symbols=symbol)
        EnvPara.FLOAT_PRECISION = torch.double
        dataset = AtomsData([atoms1, atoms2], cutoff=self.cutoff)
        self.batch_data = Batch.from_data_list(dataset)
        input_tensors = {}
        input_tensors[0] = torch.unsqueeze(self.batch_data['atomic_number'], 1).double()
        input_tensors[1] = torch.unsqueeze(self.batch_data['coordinate'], 1)
        input_tensors[2] = multi_outer_product(input_tensors[1], 2)
        self.batch_data['node_attr'] = input_tensors
        self.radial_fn = ChebyshevPoly(r_min=self.r_min, r_max=self.cutoff)
        self.cut_fn = SmoothCosineCutoff(cutoff=self.cutoff)

    def test_behler(self):
        layer = BehlerG1(n_radial=10, cut_fn=self.cut_fn,).double()
        emb1, emb2 = layer(self.batch_data).detach().numpy().reshape(2, -1)
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=6)

    def check_tensor_equivalent(self, tensor):
        E1, E2 = tensor[0][:, 0].detach().numpy().reshape(2, -1)
        np.testing.assert_array_almost_equal(E1, E2, decimal=6)
        R1, R2 = tensor[1][:, 0].detach().numpy().reshape(2, -1, 3)
        np.testing.assert_array_almost_equal(R1 @ self.R.T, R2, decimal=6)
        T1, T2 = tensor[2][:, 0].detach().numpy().reshape(2, -1, 3, 3)
        np.testing.assert_array_almost_equal(T2, self.R @ T1 @ self.R.T, decimal=6)

    def test_tensor_aggregate(self):
        layer = TensorAggregateLayer(radial_fn=self.radial_fn,
                                     cutoff_fn=self.cut_fn,
                                     n_channel=1,
                                     max_out_way=2,
                                     max_r_way=2).double()
        output_tensors = layer(self.batch_data['node_attr'], self.batch_data)
        self.check_tensor_equivalent(output_tensors)

    def test_self_interact(self):
        layer1 = SelfInteractionLayer(input_dim=1,
                                      max_in_way=2,
                                      output_dim=10).double()
        layer2 = SelfInteractionLayer(input_dim=10,
                                      max_in_way=2,
                                      output_dim=1).double()
        output_tensors = layer1(self.batch_data['node_attr'])
        output_tensors = layer2(output_tensors)
        self.check_tensor_equivalent(output_tensors)

    def test_non_linear(self):
        layer = NonLinearLayer(activate_fn=torch.sigmoid,
                               max_in_way=2).double()
        output_tensors = layer(self.batch_data['node_attr'])
        self.check_tensor_equivalent(output_tensors)

    def test_son_equivalent(self):
        layer = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                   radial_fn=self.radial_fn,
                                   cutoff_fn=self.cut_fn,
                                   max_r_way=2,
                                   max_out_way=2,
                                   input_dim=1,
                                   output_dim=1).double()
        output_tensors = layer(self.batch_data)['node_attr']
        self.check_tensor_equivalent(output_tensors)

    def test_multi_son_equivalent(self):
        layer1 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=self.radial_fn,
                                    cutoff_fn=self.cut_fn,
                                    max_r_way=2,
                                    max_out_way=2,
                                    input_dim=1,
                                    output_dim=10).double()
        layer2 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=self.radial_fn,
                                    cutoff_fn=self.cut_fn,
                                    max_r_way=2,
                                    max_out_way=2,
                                    input_dim=10,
                                    output_dim=10).double()
        layer3 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=self.radial_fn,
                                    cutoff_fn=self.cut_fn,
                                    max_r_way=2,
                                    max_out_way=2,
                                    input_dim=10,
                                    output_dim=1).double()
        output_tensors = layer3(layer2(layer1(self.batch_data)))['node_attr']
        self.check_tensor_equivalent(output_tensors)

    def test_tensor_passing_net(self):
        emb = BehlerG1(n_radial=10,
                       cut_fn=self.cut_fn,)
        model = MiaoNet(embedding_layer=emb,
                        radial_fn=self.radial_fn,
                        cutoff_fn=self.cut_fn,
                        n_layers=3,
                        max_r_way=2,
                        max_out_way=2,
                        output_dim=[10, 10, 10],
                        activate_fn=torch.sigmoid,
                        target_way=[0]).double()
        output_tensors = model(self.batch_data, ['site_energy'])['site_energy_p']
        E1, E2 = output_tensors.detach().numpy().reshape(2, -1)
        np.testing.assert_array_almost_equal(E1, E2, decimal=6)
 
    def test_tensor_passing_net_grad(self):
        emb = BehlerG1(n_radial=10,
                       cut_fn=self.cut_fn)
        radial_fn = ChebyshevPoly(r_min=self.r_min, r_max=self.cutoff)
        model = MiaoNet(embedding_layer=emb,
                        radial_fn=radial_fn,
                        cutoff_fn=self.cut_fn,
                        n_layers=3,
                        max_r_way=2,
                        max_out_way=2,
                        output_dim=[10, 10, 10],
                        activate_fn=torch.sigmoid,
                        target_way=[0]).double()
        output_tensors = model(self.batch_data, ['forces'])['forces_p']
        F1, F2 = output_tensors.detach().numpy().reshape(2, -1, 3)
        np.testing.assert_array_almost_equal(F1 @ self.R.T, F2, decimal=6)


if __name__ == '__main__':
    unittest.main()
