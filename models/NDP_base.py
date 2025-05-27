import torch
import torch.nn as nn
from .rigid_body import _6d_to_SO3, euler_to_SO3, quaternion_to_SO3, exp_se3, exp_so3, _copysign
# from .position_encoding import *
import torch.nn.functional as F
from torch.autograd.functional import jacobian
import numpy as np
from collections import OrderedDict



class Deformation_Pyramid ():

    def __init__(self, depth, width, device, k0, m, rotation_format, nonrigidity_est=False, motion='SE3'):

        pyramid = []


        assert motion in [ "Sim3", "SE3", "sflow"]


        for i in range (m):
            pyramid.append(
                NDPLayer(depth,
                         width,
                         k0,
                         i+1,
                         rotation_format,
                         nonrigidity_est=nonrigidity_est & (i!=0),
                         motion=motion
                         ).to(device)
            )


        self.pyramid = pyramid
        self.n_hierarchy = m

    def warp(self, x, max_level=None, min_level=0):

        if max_level is None:
            max_level = self.n_hierarchy - 1

        assert max_level < self.n_hierarchy, "more level than defined"

        data = {}

        for i in range(min_level, max_level + 1):
            x, nonrigidity = self.pyramid[i](x)
            data[i] = (x, nonrigidity)
        return x, data

    def gradient_setup(self, optimized_level):

        assert optimized_level < self.n_hierarchy, "more level than defined"

        # optimize current level, freeze the other levels
        for i in range( self.n_hierarchy):
            net = self.pyramid[i]
            if i == optimized_level:
                for param in net.parameters():
                    param.requires_grad = True
            else:
                for param in net.parameters():
                    param.requires_grad = False



class NDPLayer(nn.Module):
    def __init__(self, depth, width, k0, m, rotation_format="euler", nonrigidity_est=False, motion='SE3'):
        super().__init__()

        self.k0 = k0
        self.m = m
        dim_x =  6
        self.nonrigidity_est = nonrigidity_est
        self.motion = motion
        self.input= nn.Sequential( nn.Linear(dim_x,width), nn.LeakyReLU())
        self.mlp = MLP(depth=depth,width=width)

        self.rotation_format = rotation_format


        """rotation branch"""
        if self.motion in [ "Sim3", "SE3"] :

            if self.rotation_format in [ "axis_angle", "euler" ]:
                self.rot_brach = nn.Linear(width, 3)
            elif self.rotation_format == "quaternion":
                self.rot_brach = nn.Linear(width, 4)
            elif self.rotation_format == "6D":
                self.rot_brach = nn.Linear(width, 6)


            if self.motion == "Sim3":
                self.s_branch = nn.Linear(width, 1) # scale branch


        """translation branch"""
        self.trn_branch = nn.Linear(width, 3)


        """rigidity branch"""
        if self.nonrigidity_est:
            self.nr_branch = nn.Linear(width, 1)
            self.sigmoid = nn.Sigmoid()


        # Apply small scaling on the MLP output, s.t. the optimization can start from near identity pose
        self.mlp_scale = 0.001

        self._reset_parameters()

    def forward (self, x):

        fea = self.posenc( x )
        fea = self.input(fea)
        fea = self.mlp(fea)

        t = self.mlp_scale * self.trn_branch ( fea )

        if self.motion == "SE3":
            R = self.get_Rotation(fea)
            x_ = (R @ x[..., None]).squeeze() + t

        elif self.motion == "Sim3":
            R = self.get_Rotation(fea)
            s = self.mlp_scale * self.s_branch(fea) + 1  # optimization starts with scale==1
            x_ = s * (R @ x[..., None]).squeeze() + t

        else: # scene flow
            x_ = x + t


        if self.nonrigidity_est:
            nonrigidity =self.sigmoid( self.mlp_scale * self.nr_branch(fea) )
            x_ = x + nonrigidity * (x_ - x)
            nonrigidity = nonrigidity.squeeze()
        else:
            nonrigidity = None


        return x_.squeeze(), nonrigidity



    def get_Rotation (self, fea):

        R = self.mlp_scale * self.rot_brach( fea )

        if self.rotation_format == "euler":
            R = euler_to_SO3(R)
        elif self.rotation_format == "axis_angle":
            theta = torch.norm(R, dim=-1, keepdim=True)
            w = R / theta
            R = exp_so3(w, theta)
        elif self.rotation_format =='quaternion':
            s = (R * R).sum(1)
            R = R / _copysign(torch.sqrt(s), R[:, 0])[:, None]
            R = quaternion_to_SO3(R)
        elif self.rotation_format == "6D":
            R = _6d_to_SO3(R)

        return R


    def posenc(self, pos):
        pi = 3.14
        x_position, y_position, z_position = pos[..., 0:1], pos[..., 1:2], pos[..., 2:3]
        # mul_term = ( 2 ** (torch.arange(self.m, device=pos.device).float() + self.k0) * pi ).reshape(1, -1)
        mul_term = (2 ** (self.m + self.k0)  )#.reshape(1, -1)

        sinx = torch.sin(x_position * mul_term)
        cosx = torch.cos(x_position * mul_term)
        siny = torch.sin(y_position * mul_term)
        cosy = torch.cos(y_position * mul_term)
        sinz = torch.sin(z_position * mul_term)
        cosz = torch.cos(z_position * mul_term)
        pe = torch.cat([sinx, cosx, siny, cosy, sinz, cosz], dim=-1)
        return pe


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)



class Nerfies_Deformation(nn.Module):
    '''
    Our re-implementation of the deformation model from [Nerfies, ICCV'21], https://arxiv.org/abs/2011.12948
    '''
    def __init__(self, depth=7, width=128, max_iter = 5000):
        super().__init__()

        self.k0 = -3
        self.m=6
        dim_x = self.m * 6 + 3
        self.input= nn.Sequential( nn.Linear(dim_x,width), nn.LeakyReLU())
        self.mlp = MLP(depth=depth,width=width)
        self.w_branch = nn.Linear(width, 3)
        self.v_branch = nn.Linear(width, 3)

        self.max_iter = max_iter
        self.N = 0.6 * max_iter



    def forward(self, x, iter ):
        warpped_x = self.warp(x, iter)
        J = self.batched_jacobian(self.warp, x, iter)
        return warpped_x, J


    def batched_jacobian(self, f, x, iter):
        f_sum = lambda x: torch.sum(f(x, iter), axis=0)
        return jacobian(f_sum, x).transpose(0,1)


    def posenc(self, pos, iter):

        pi = 3.14

        # sliding window
        a = self.m * iter / self.N
        w_a = ( 1 - torch.cos( torch.clamp(a-torch.arange(self.m, device=pos.device).float(), min=0, max=1) * pi ) ) / 2
        w_a = w_a[None]

        x_position, y_position, z_position = pos[..., 0:1], pos[..., 1:2], pos[..., 2:3]
        mul_term = (
                2 ** (torch.arange(self.m, device=pos.device).float() + self.k0) * pi
                ).reshape(1, -1)

        sinx = torch.sin(x_position * mul_term) * w_a
        cosx = torch.cos(x_position * mul_term) * w_a
        siny = torch.sin(y_position * mul_term) * w_a
        cosy = torch.cos(y_position * mul_term) * w_a
        sinz = torch.sin(z_position * mul_term) * w_a
        cosz = torch.cos(z_position * mul_term) * w_a
        position_code = torch.cat([sinx, cosx, siny, cosy, sinz, cosz], dim=-1)
        position_code = torch.cat( [pos, position_code], dim= -1)
        return position_code

    def warp (self, x, iter) :
        fea = self.posenc(x, iter)
        fea = self.input(fea)
        fea = self.mlp(fea)
        w = self.w_branch(fea)
        v = self.v_branch(fea)
        theta = torch.norm(w, dim=-1, keepdim=True)
        w = w/theta
        v = v/theta
        R, t = exp_se3(w, v, theta)
        _x = ( R @ x[..., None] + t ).squeeze()
        return _x.squeeze()


class Neural_Prior(torch.nn.Module):
    '''
    Borrow from [Neural Scene flow Prior, NIPS'21], https://arxiv.org/abs/2111.01253
    '''
    def __init__(self, dim_x=3, filter_size=128, act_fn='relu'):
        super().__init__()
        # input layer (default: xyz -> 128)
        self.layer1 = torch.nn.Linear(dim_x, filter_size)
        # hidden layers (default: 128 -> 128)
        self.layer2 = torch.nn.Linear(filter_size, filter_size)
        self.layer3 = torch.nn.Linear(filter_size, filter_size)
        self.layer4 = torch.nn.Linear(filter_size, filter_size)
        self.layer5 = torch.nn.Linear(filter_size, filter_size)
        self.layer6 = torch.nn.Linear(filter_size, filter_size)
        self.layer7 = torch.nn.Linear(filter_size, filter_size)
        self.layer8 = torch.nn.Linear(filter_size, filter_size)
        # output layer (default: 128 -> 3)
        self.layer9 = torch.nn.Linear(filter_size, 3)

        # activation functions
        if act_fn == 'relu':
            self.act_fn = torch.nn.functional.relu
        elif act_fn == 'sigmoid':
            self.act_fn = torch.nn.functional.sigmoid

    def forward(self, x):
        x = self.act_fn(self.layer1(x))
        x = self.act_fn(self.layer2(x))
        x = self.act_fn(self.layer3(x))
        x = self.act_fn(self.layer4(x))
        x = self.act_fn(self.layer5(x))
        x = self.act_fn(self.layer6(x))
        x = self.act_fn(self.layer7(x))
        x = self.act_fn(self.layer8(x))
        x = self.layer9(x)

        return x


class MLP(torch.nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        self.pts_linears = nn.ModuleList( [nn.Linear(width, width) for i in range(depth - 1)])

    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            x = F.relu(x)
        return x
    


# @title Define SIREN deformation model
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords#.requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output #, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
    

    