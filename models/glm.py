import torch
import torch.nn as nn
from NDNT import NDN 
import NDNT.modules.layers as layers
from NDNT.metrics.poisson_loss import PoissonLoss_datafilter
import numpy as np
from torchvision.transforms import CenterCrop
from models.base import Encoder

class GQM(Encoder):

    def __init__(self, input_dims,
            NC=None,
            nquad=1,
            reg_vals={'d2x':1e-5, 'd2t':1e-5},
            cids=None,
            time_embed=False,
            **kwargs):
        
        super().__init__(input_dims,NC=NC,cids=cids,**kwargs)

        if cids is None:
            self.cids = list(range(NC))
        else:
            self.cids = cids
            NC = len(cids)
        
        self.input_dims = input_dims
        self.name = 'GQM_{}q-{}'.format(nquad,np.asarray(cids)).replace('[', '').replace(']','')

        self.core = nn.Sequential()

        lin = layers.NDNLayer(input_dims=input_dims,
            num_filters=NC,
            NLtype='lin',
            norm_type=0,
            bias=False,
            reg_vals = reg_vals)
        
        self.core.add_module('lin0', lin)

        for q in range(nquad):
            quad = layers.NDNLayer(input_dims=input_dims,
                num_filters=NC,
                NLtype='square',
                norm_type=0,
                bias=False,
                reg_vals = reg_vals)

            self.core.add_module('quad{}'.format(q), quad)

        self.core_subunits = NC*nquad + NC
        self.nquad = nquad
        self.time_embed = time_embed
        self.num_lags = input_dims[-1]

        self.bias = torch.nn.Parameter(torch.zeros(NC))
        self.output_NL = nn.Softplus()

        self.loss = PoissonLoss_datafilter()
        self.loss.unit_normalization = False

    def forward(self, data):

        if self.time_embed:
            iix = np.maximum(np.arange(data['stim'].shape[0])[:,None]-np.arange(self.input_dims[-1])[None,:], 0)
            stim = data['stim'].reshape([-1] + self.input_dims[:-1]).permute(1,2,3,0)[...,iix].permute(3,0,1,2,4)
            stim = torch.flatten(stim, start_dim=1)
        else:
            stim = data['stim']

        x = 0
        for layer in self.core:
            x += layer(stim)

        use_offsets = hasattr(self, "offsets")
        use_gains = hasattr(self, "gains")

        if use_offsets:
            offset = self.offval
            for offmod,stim in zip(self.offsets, self.offsetstims):
                offset = offset + offmod(data[stim])

        if use_gains:
            gain = self.gainval
            for gainmod,stim in zip(self.gains, self.gainstims):
                gain = gain * (self.gainval + gainmod(data[stim]))

        if self.modify:
            x *= gain
            x += offset

        x += self.bias[None,:]
        
        return self.output_NL(x)

class GLM(GQM):

    def __init__(self, input_dims,
            NC=None,
            nquad=0,
            **kwargs):
        
        if nquad > 0:
            nquad = 0

        super().__init__(input_dims, NC=NC, nquad=nquad, **kwargs)
        self.name = 'GLM-{}'.format(np.asarray(self.cids)).replace('[', '').replace(']','')

class EnergyModel(Encoder):

    def __init__(self, input_dims,
            NC=None,
            nquad=1,
            reg_vals={'d2x':1e-5, 'd2t':1e-5},
            cids=None,
            time_embed=False,
            **kwargs):
        
        super().__init__(input_dims,NC=NC,cids=cids,**kwargs)

        if cids is None:
            self.cids = list(range(NC))
        else:
            self.cids = cids
            NC = len(cids)
        
        self.input_dims = input_dims
        self.name = 'EnergyModel'

        self.core = nn.Sequential()
        self.register_buffer('excitatory' , torch.cat( (torch.ones(nquad), -torch.ones(nquad))))

        for q in range(nquad):
            quad = layers.NDNLayer(input_dims=input_dims,
                num_filters=NC,
                NLtype='square',
                norm_type=2,
                bias=False,
                reg_vals = reg_vals)

            self.core.add_module('ex_quad{}'.format(q), quad)
        
        for q in range(nquad):
            quad = layers.NDNLayer(input_dims=input_dims,
                num_filters=NC,
                NLtype='square',
                norm_type=2,
                bias=False,
                reg_vals = reg_vals)

            self.core.add_module('sup_quad{}'.format(q), quad)

        self.core_subunits = NC*nquad*2
        self.nquad = nquad
        self.time_embed = time_embed
        self.num_lags = input_dims[-1]

        self.bias = torch.nn.Parameter(torch.zeros(NC))
        self.output_NL = nn.Softplus()

        self.loss = PoissonLoss_datafilter()
        self.loss.unit_normalization = False

    def forward(self, data):

        if self.time_embed:
            iix = np.maximum(np.arange(data['stim'].shape[0])[:,None]-np.arange(self.input_dims[-1])[None,:], 0)
            stim = data['stim'].reshape([-1] + self.input_dims[:-1]).permute(1,2,3,0)[...,iix].permute(3,0,1,2,4)
            stim = torch.flatten(stim, start_dim=1)
        else:
            stim = data['stim']

        x = 0
        for i, layer in enumerate(self.core):
            x += self.excitatory[i]*layer(stim)

        use_offsets = hasattr(self, "offsets")
        use_gains = hasattr(self, "gains")

        if use_offsets:
            offset = self.offval
            for offmod,stim in zip(self.offsets, self.offsetstims):
                offset = offset + offmod(data[stim])

        if use_gains:
            gain = self.gainval
            for gainmod,stim in zip(self.gains, self.gainstims):
                gain = gain * (self.gainval + gainmod(data[stim]))

        if self.modify:
            x *= gain
            x += offset

        x += self.bias[None,:]
        
        return self.output_NL(x)
    
class freqGLM(GQM):

    def __init__(self, input_dims,
            NC=None,
            nquad=0,
            **kwargs):
        
        if nquad > 0:
            nquad = 0

        super().__init__(input_dims, NC=NC, nquad=nquad, **kwargs)
        self.name = 'freqGLM-{}'.format(np.asarray(self.cids)).replace('[', '').replace(']','')
        self.pad_size = self.input_dims[1:-1]*2
        self.crop_inds = [(int(s/4), int(s-s/4)) for s in self.pad_size]
    
    def forward(self, data):

        if self.time_embed:
            iix = np.maximum(np.arange(data['stim'].shape[0])[:,None]-np.arange(self.input_dims[-1])[None,:], 0)
            stim = data['stim'].reshape([-1] + self.input_dims[:-1]).permute(1,2,3,0)[...,iix].permute(3,0,1,2,4)
            
        else:
            stim = data['stim']
        
        fstim = torch.fft.fftn(stim, s=self.pad_size)
        fstim = torch.fft.fftshift(fstim, dim=2)
        fstim = fstim.real**2
        fstim = fstim[..., self.crop_inds[0][0]:self.crop_inds[0][1], self.crop_inds[1][0]:self.crop_inds[1][1], self.crop_inds[2][0]:self.crop_inds[2][1]]

        stim = torch.flatten(fstim, start_dim=1)

        x = 0
        for layer in self.core:
            x += layer(stim)/1000

        x += self.bias[None,:]

        return self.output_NL(x)

class sGQM(Encoder):

    def __init__(self,
            input_dims,
            num_subunits=10,
            NC=None,
            nquad=2,
            reg_vals={'d2x':0.001, 'd2t':0.001},
            reg_readout={'l2':0.001},
            cids=None,
            **kwargs):
        
        super().__init__(input_dims,NC=NC,cids=cids,**kwargs)

        if cids is None:
            self.cids = list(range(NC))
        else:
            self.cids = cids
            NC = len(cids)
        
        self.input_dims = input_dims
        self.name = 'sGQM_{}q-{}'.format(nquad,np.asarray(cids)).replace('[', '').replace(']','')

        self.core = nn.Sequential()

        lin = layers.NDNLayer(input_dims=input_dims,
            num_filters=num_subunits,
            NLtype='lin',
            norm_type=1,
            bias=False,
            reg_vals = reg_vals)
        
        self.core.add_module('basis', lin)

        self.readout = nn.ModuleList()
        lin_readout = layers.NDNLayer(input_dims=[num_subunits, 1, 1, 1],
            num_filters=NC,
            NLtype='lin',
            norm_type=2,
            bias=False,
            reg_vals = reg_readout)
        
        self.readout.append(lin_readout)
        

        for q in range(nquad):
            quad = layers.NDNLayer(
                input_dims=[num_subunits, 1, 1, 1],
                num_filters=NC,
                NLtype='square',
                norm_type=2,
                bias=False,
                reg_vals = reg_readout)

            self.readout.add_module('quad{}'.format(q), quad)

        self.core_subunits = NC*nquad + NC
        self.nquad = nquad
        self.num_lags = input_dims[-1]
        
        # self.alpha = torch.nn.Parameter(torch.ones(NC), requires_grad=False)
        # self.baseline = torch.nn.Parameter(torch.zeros(NC), requires_grad=False)

        self.bias = torch.nn.Parameter(torch.zeros(NC))
        
        self.output_NL = nn.Softplus()


    def forward(self, data):

        stim = self.core(data['stim'])

        x = 0
        for layer in self.readout:
            x += layer(stim)

        use_offsets = hasattr(self, "offsets")
        use_gains = hasattr(self, "gains")

        if use_offsets:
            offset = self.offval
            for offmod,stim in zip(self.offsets, self.offsetstims):
                offset = offset + offmod(data[stim])

        if use_gains:
            gain = self.gainval
            for gainmod,stim in zip(self.gains, self.gainstims):
                gain = gain * (self.gainval + gainmod(data[stim]))

        if self.modify:
            x *= gain
            x += offset

        x += self.bias[None,:]
        
        return self.output_NL(x)