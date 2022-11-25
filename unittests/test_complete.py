from scipy.io import loadmat
from scipy.stats import special_ortho_group
from ase import Atoms
from ase.io import read, write
from einops import repeat
from tensornet.layer.embedding import OneHot, BehlerG1
from tensornet.layer.cutoff import SmoothCosineCutoff
from tensornet.layer.equivalent import SOnEquivalentLayer, TensorAggregateLayer, SelfInteractionLayer, NonLinearLayer
from tensornet.layer.radial import ChebyshevPoly
from tensornet.models import TensorMessagePassingNet
from tensornet.utils import setup_seed, multi_outer_product
from tensornet.data import convert_frames
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
        self.batch_data = convert_frames([a1, a2], cutoff=self.cutoff)

    def test_behler(self):
        cut_fn = SmoothCosineCutoff(cutoff=self.cutoff)
        atomic_fn = OneHot([1])
        layer = BehlerG1(n_radial=10,
                         cut_fn=cut_fn, 
                         atomic_fn=atomic_fn,)
        emb = layer(self.batch_data).detach().numpy()
        np.testing.assert_array_almost_equal(emb[0, 0], emb[1, 0], decimal=6)

    def test_son_0(self):
        cut_fn = SmoothCosineCutoff(cutoff=self.cutoff)
        atomic_fn = OneHot([1])
        emb = BehlerG1(n_radial=10,
                       cut_fn=cut_fn, 
                       atomic_fn=atomic_fn,)
        radial_fn = ChebyshevPoly(r_min=self.r_min, r_max=self.cutoff)
        layer1 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=radial_fn,
                                    cutoff_fn=cut_fn,
                                    max_r_way=0,
                                    max_out_way=0,
                                    input_dim=emb.n_channel,
                                    output_dim=10)
        layer2 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=radial_fn,
                                    cutoff_fn=cut_fn,
                                    max_r_way=0,
                                    max_out_way=0,
                                    input_dim=emb.n_channel,
                                    output_dim=10)
        input_tensors = {0: emb(self.batch_data)}
        output_tensors = layer1(input_tensors, 
                                self.batch_data)
        e = output_tensors[0].detach().numpy()
        output_tensors = layer2(output_tensors,
                                self.batch_data)[0].detach().numpy()
        self.assertFalse(np.allclose(output_tensors[0, 0], output_tensors[1, 0]))
        
    def test_son_1(self):
        cut_fn = SmoothCosineCutoff(cutoff=self.cutoff)
        atomic_fn = OneHot([1])
        emb = BehlerG1(n_radial=10,
                       cut_fn=cut_fn, 
                       atomic_fn=atomic_fn,)
        radial_fn = ChebyshevPoly(r_min=self.r_min, r_max=self.cutoff)
        layer1 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=radial_fn,
                                    cutoff_fn=cut_fn,
                                    max_r_way=1,
                                    max_out_way=1,
                                    input_dim=emb.n_channel,
                                    output_dim=10)
        layer2 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=radial_fn,
                                    cutoff_fn=cut_fn,
                                    max_r_way=1,
                                    max_out_way=1,
                                    input_dim=emb.n_channel,
                                    output_dim=10)
        input_tensors = {0: emb(self.batch_data)}
        output_tensors = layer1(input_tensors, 
                                self.batch_data)
        output_tensors = layer2(output_tensors, 
                                self.batch_data)[0].detach().numpy()
        self.assertFalse(np.allclose(output_tensors[0, 0], output_tensors[1, 0]))
    
    def test_son_2(self):
        cut_fn = SmoothCosineCutoff(cutoff=self.cutoff)
        atomic_fn = OneHot([1])
        emb = BehlerG1(n_radial=10,
                       cut_fn=cut_fn, 
                       atomic_fn=atomic_fn,)
        radial_fn = ChebyshevPoly(r_min=self.r_min, r_max=self.cutoff)
        layer1 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=radial_fn,
                                    cutoff_fn=cut_fn,
                                    max_r_way=2,
                                    max_out_way=2,
                                    input_dim=emb.n_channel,
                                    output_dim=10)
        layer2 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radial_fn=radial_fn,
                                    cutoff_fn=cut_fn,
                                    max_r_way=2,
                                    max_out_way=2,
                                    input_dim=emb.n_channel,
                                    output_dim=10)
        input_tensors = {0: emb(self.batch_data)}
        output_tensors = layer1(input_tensors, 
                                self.batch_data)
        output_tensors = layer2(output_tensors, 
                                self.batch_data)[0].detach().numpy()
        self.assertFalse(np.allclose(output_tensors[0, 0], output_tensors[1, 0]))


if __name__ == '__main__':
    setup_seed(0)
    unittest.main()
