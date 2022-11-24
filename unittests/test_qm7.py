from scipy.io import loadmat
from scipy.stats import special_ortho_group
from ase import Atoms
from ase.io import read
from einops import repeat
from tensornet.layer.embedding import OneHot, BehlerG1
from tensornet.layer.cutoff import SmoothCosineCutoff
from tensornet.layer.equivalent import SOnEquivalentLayer, TensorAggregateLayer, SelfInteractionLayer, NonLinearLayer
from tensornet.layer.radius import ChebyshevPoly
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
        self.r_min = 1.5
        self.cutoff = 3.5

        dim = 3
        self.R = special_ortho_group.rvs(dim=dim)

        datapath = '../dataset/qm7.mat'
        raw_data = loadmat(datapath)
        coordinate = raw_data['R'][0][:6]
        symbol = raw_data['Z'][0][:6]
        atoms1 = Atoms(positions=coordinate, symbols=symbol)
        atoms2 = Atoms(positions=coordinate @ self.R.T, symbols=symbol)
        self.batch_data = convert_frames([atoms1, atoms2], self.cutoff)

    def test_behler(self):
        cut_fn = SmoothCosineCutoff(cutoff=self.cutoff)
        atomic_fn = OneHot([0, 1, 6, 7, 8, 16])
        layer = BehlerG1(n_radius=10,
                        cut_fn=cut_fn, 
                        atomic_fn=atomic_fn,)
        emb = layer.forward(coordinate=self.batch_data['coordinate'], 
                            atomic_number=self.batch_data['symbol'], 
                            neighbor=self.batch_data['neighbor'],
                            mask=self.batch_data['mask']).detach().numpy()
        np.testing.assert_array_almost_equal(emb[0], emb[1], decimal=6)
    
    def check_tensor_equivalent(self, tensor):
        E1 = tensor[0][0].detach().numpy()
        E2 = tensor[0][1].detach().numpy()
        np.testing.assert_array_almost_equal(E1, E2, decimal=5)
        R1 = tensor[1][0, :, 0].detach().numpy()
        R2 = tensor[1][1, :, 0].detach().numpy()
        np.testing.assert_array_almost_equal(R1 @ self.R.T, R2, decimal=5)
        T1 = tensor[2][0, :, 0].detach().numpy()
        T2 = tensor[2][1, :, 0].detach().numpy()
        np.testing.assert_array_almost_equal(T2, self.R @ T1 @ self.R.T, decimal=4)

    def test_tensor_aggregate(self):
        input_tensors = {}
        input_tensors[1] = torch.unsqueeze(self.batch_data['coordinate'], 2)
        radius_fn = ChebyshevPoly
        radius_fn_para = {'r_min': self.r_min, 'r_max': self.cutoff}
        layer = TensorAggregateLayer(radius_fn=radius_fn,
                                     radius_fn_para=radius_fn_para,
                                     max_out_way=2,
                                     max_r_way=2)
        output_tensors = layer(input_tensors, 
                               self.batch_data['coordinate'],
                               self.batch_data['neighbor'],
                               self.batch_data['mask'],
                               None, None)
        self.check_tensor_equivalent(output_tensors)

    def test_self_interact(self):
        input_tensors = {}
        input_tensors[0] = torch.unsqueeze(self.batch_data['symbol'], 2).float()
        input_tensors[1] = torch.unsqueeze(self.batch_data['coordinate'], 2)
        input_tensors[2] = multi_outer_product(input_tensors[1], 2)
        layer1 = SelfInteractionLayer(input_dim=1,
                                      max_in_way=2,
                                      output_dim=10)
        layer2 = SelfInteractionLayer(input_dim=10,
                                      max_in_way=2,
                                      output_dim=1)
        output_tensors = layer1(input_tensors)
        output_tensors = layer2(output_tensors)
        self.check_tensor_equivalent(output_tensors)
    
    def test_non_linear(self):
        input_tensors = {}
        input_tensors[0] = torch.unsqueeze(self.batch_data['symbol'], 2).float()
        input_tensors[1] = torch.unsqueeze(self.batch_data['coordinate'], 2)
        input_tensors[2] = multi_outer_product(input_tensors[1], 2)
        layer = NonLinearLayer(activate_fn=torch.sigmoid,
                               max_in_way=2)
        output_tensors = layer(input_tensors)
        self.check_tensor_equivalent(output_tensors)

    def test_son_equivalent(self):
        input_tensors = {}
        input_tensors[0] = torch.unsqueeze(self.batch_data['symbol'], 2).float()
        input_tensors[1] = torch.unsqueeze(self.batch_data['coordinate'], 2)
        input_tensors[2] = multi_outer_product(input_tensors[1], 2)
        radius_fn = ChebyshevPoly
        radius_fn_para = {'r_min': self.r_min, 'r_max': self.cutoff}
        layer = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                   radius_fn=radius_fn,
                                   radius_fn_para=radius_fn_para,
                                   max_r_way=2,
                                   max_out_way=2,
                                   input_dim=1,
                                   output_dim=1)
        output_tensors = layer(input_tensors, 
                               self.batch_data['coordinate'],
                               self.batch_data['neighbor'],
                               self.batch_data['mask'],
                               None, None)
        self.check_tensor_equivalent(output_tensors)
    
    def test_multi_son_equivalent(self):
        input_tensors = {}
        input_tensors[1] = torch.unsqueeze(self.batch_data['coordinate'], 2)
        radius_fn = ChebyshevPoly
        radius_fn_para = {'r_min': self.r_min, 'r_max': self.cutoff}
        layer1 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radius_fn=radius_fn,
                                    radius_fn_para=radius_fn_para,
                                    max_r_way=2,
                                    max_out_way=2,
                                    input_dim=1,
                                    output_dim=10)
        layer2 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radius_fn=radius_fn,
                                    radius_fn_para=radius_fn_para,
                                    max_r_way=2,
                                    max_out_way=2,
                                    input_dim=10,
                                    output_dim=10)
        layer3 = SOnEquivalentLayer(activate_fn=torch.sigmoid,
                                    radius_fn=radius_fn,
                                    radius_fn_para=radius_fn_para,
                                    max_r_way=2,
                                    max_out_way=2,
                                    input_dim=10,
                                    output_dim=1)

        output_tensors = layer1(input_tensors, 
                                self.batch_data['coordinate'],
                                self.batch_data['neighbor'],
                                self.batch_data['mask'],
                                None, None)
        output_tensors = layer2(output_tensors, 
                                self.batch_data['coordinate'],
                                self.batch_data['neighbor'],
                                self.batch_data['mask'],
                                None, None)
        output_tensors = layer3(output_tensors, 
                                self.batch_data['coordinate'],
                                self.batch_data['neighbor'],
                                self.batch_data['mask'],
                                None, None)
        self.check_tensor_equivalent(output_tensors)
        
    def test_tensor_passing_net(self):
        cut_fn = SmoothCosineCutoff(cutoff=self.cutoff)
        atomic_fn = OneHot([0, 1, 6, 7, 8, 16])
        emb = BehlerG1(n_radius=10,
                       cut_fn=cut_fn, 
                       atomic_fn=atomic_fn,)
        radius_fn = ChebyshevPoly
        radius_fn_para = {'r_min': self.r_min, 'r_max': self.cutoff}
        model = TensorMessagePassingNet(embedding_layer=emb,
                                        radius_fn=radius_fn,
                                        radius_fn_para=radius_fn_para,
                                        n_layers=3,
                                        max_r_way=2,
                                        max_out_way=2,
                                        output_dim=[10, 10, 10],
                                        activate_fn=torch.sigmoid,
                                        target_way=[0])
        output_tensors = model(self.batch_data['coordinate'],
                               self.batch_data['symbol'],
                               self.batch_data['neighbor'],
                               self.batch_data['mask'],
                               None, None)
        E1 = output_tensors[0][0].detach().numpy()
        E2 = output_tensors[0][1].detach().numpy()
        np.testing.assert_array_almost_equal(E1, E2, decimal=5)

    def test_tensor_passing_net_grad(self):
        cut_fn = SmoothCosineCutoff(cutoff=self.cutoff)
        atomic_fn = OneHot([0, 1, 6, 7, 8, 16])
        emb = BehlerG1(n_radius=10,
                       cut_fn=cut_fn, 
                       atomic_fn=atomic_fn,)
        radius_fn = ChebyshevPoly
        radius_fn_para = {'r_min': self.r_min, 'r_max': self.cutoff}
        model = TensorMessagePassingNet(embedding_layer=emb,
                                        radius_fn=radius_fn,
                                        radius_fn_para=radius_fn_para,
                                        n_layers=3,
                                        max_r_way=2,
                                        max_out_way=2,
                                        output_dim=[10, 10, 10],
                                        activate_fn=torch.sigmoid,
                                        target_way=[0])
        self.batch_data['coordinate'].requires_grad_()
        output_tensors = model(self.batch_data['coordinate'],
                               self.batch_data['symbol'],
                               self.batch_data['neighbor'],
                               self.batch_data['mask'],
                               None, None)
        forces = -torch.autograd.grad(
            output_tensors[0].sum(),
            self.batch_data['coordinate'],
            create_graph=True,
            retain_graph=True
        )[0]
        F1 = forces[0, :].detach().numpy()
        F2 = forces[1, :].detach().numpy()
        np.testing.assert_array_almost_equal(F1 @ self.R.T, F2, decimal=4)
        

if __name__ == '__main__':
    unittest.main()