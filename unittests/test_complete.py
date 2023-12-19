from ase import Atoms
from hotpp.layer import BehlerG1, SmoothCosineCutoff, SOnEquivalentLayer, ChebyshevPoly
from hotpp.utils import setup_seed
from hotpp.data import AtomsData
from torch_geometric.data import Batch
import unittest
import numpy as np
import torch


class TestMolecule(unittest.TestCase):
    def setUp(self):
        setup_seed(0)

        self.max_in_way = 2        # 0: scalar  1: vector  2: matrix  ...
        self.r_min = 0.2
        self.cutoff = 3.5

        a1 = Atoms(symbols='H5',
                   positions=[[0.,  0., 0.],
                              [0., -1., 0.],
                              [0.,  1., 0.],
                              [0.5, -np.sqrt(3) / 2, 0.],
                              [0.5,  np.sqrt(3) / 2, 0.]])
        a2 = Atoms(symbols='H5',
                   positions=[[0.,  0., 0.],
                              [0., -1., 0.],
                              [0.,  1., 0.],
                              [ 0.5, -np.sqrt(3) / 2, 0.],
                              [-0.5, -np.sqrt(3) / 2, 0.]])
        dataset = AtomsData([a1, a2], cutoff=self.cutoff)
        self.batch_data = Batch.from_data_list(dataset)
        self.cut_fn = SmoothCosineCutoff(cutoff=self.cutoff)
        self.radial_fn = ChebyshevPoly(r_min=self.r_min, r_max=self.cutoff)

    def test_behler(self):
        
        layer = BehlerG1(n_radial=10,
                         cut_fn=self.cut_fn,)
        emb = layer(self.batch_data).detach().numpy()
        np.testing.assert_array_almost_equal(emb[0], emb[5], decimal=6)

    def test_son_0(self):
        emb = BehlerG1(n_radial=10, cut_fn=self.cut_fn, )
        layer1 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=self.radial_fn,
                                    cutoff_fn=self.cut_fn,
                                    max_r_way=0,
                                    max_out_way=0,
                                    input_dim=emb.n_channel,
                                    output_dim=10)
        layer2 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=self.radial_fn,
                                    cutoff_fn=self.cut_fn,
                                    max_r_way=0,
                                    max_out_way=0,
                                    input_dim=emb.n_channel,
                                    output_dim=10)
        self.batch_data['node_attr'] = {0: emb(self.batch_data)}
        output_tensors = layer1(self.batch_data)
        output_tensors = layer2(self.batch_data)['node_attr'][0].detach().numpy()
        self.assertFalse(np.allclose(output_tensors[0], output_tensors[5]))

    def test_son_1(self):
        emb = BehlerG1(n_radial=10, cut_fn=self.cut_fn, )
        layer1 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=self.radial_fn,
                                    cutoff_fn=self.cut_fn,
                                    max_r_way=1,
                                    max_out_way=1,
                                    input_dim=emb.n_channel,
                                    output_dim=10)
        layer2 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=self.radial_fn,
                                    cutoff_fn=self.cut_fn,
                                    max_r_way=1,
                                    max_out_way=1,
                                    input_dim=emb.n_channel,
                                    output_dim=10)
        self.batch_data['node_attr'] = {0: emb(self.batch_data)}
        output_tensors = layer1(self.batch_data)
        output_tensors = layer2(self.batch_data)['node_attr'][0].detach().numpy()
        self.assertFalse(np.allclose(output_tensors[0], output_tensors[5]))
    
    def test_son_2(self):
        emb = BehlerG1(n_radial=10, cut_fn=self.cut_fn, )
        layer1 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=self.radial_fn,
                                    cutoff_fn=self.cut_fn,
                                    max_r_way=2,
                                    max_out_way=2,
                                    input_dim=emb.n_channel,
                                    output_dim=10)
        layer2 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=self.radial_fn,
                                    cutoff_fn=self.cut_fn,
                                    max_r_way=2,
                                    max_out_way=2,
                                    input_dim=emb.n_channel,
                                    output_dim=10)
        self.batch_data['node_attr'] = {0: emb(self.batch_data)}
        output_tensors = layer1(self.batch_data)
        output_tensors = layer2(self.batch_data)['node_attr'][0].detach().numpy()
        self.assertFalse(np.allclose(output_tensors[0], output_tensors[5]))


if __name__ == '__main__':
    setup_seed(0)
    unittest.main()
