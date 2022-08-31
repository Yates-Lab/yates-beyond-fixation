import torch
import torch.nn as nn

from NDNT.modules import layers as layers
from NDNT.utils.NDNutils import initialize_gaussian_envelope
from models.base import Encoder

def Shifter():
    return nn.Sequential(
                nn.Linear(2, 20, bias=False),
                nn.Softplus(),
                nn.Linear(20, 2, bias=True))

def Shifter_with_time(num_lags):
    return nn.Sequential(
                nn.Linear(2, 20, bias=False),
                nn.Softplus(),
                nn.Linear(20, 2, bias=True),
                nn.Linear(2, num_lags, bias=False),
                nn.Softplus(),
                nn.Linear(num_lags, num_lags, bias=True))
                
class ShifterModel(Encoder):

    def __init__(self, input_dims,
            NC=None,
            num_subunits=[16, 8, 4],
            filter_width=[9, 5, 5],
            reg_vals={'d2x':0.001, 'd2t':0.001, 'center':0.01},
            reg_readout={'l2':0.001},
            reg_hidden={'l2':0.1},
            NLtype='relu',
            noise_sigma=0,
            norm_type=1,
            modifiers=None,
            cids=None,
            shifter=True,
            drifter=False,
            *args, **kwargs):
        
        super().__init__(input_dims=input_dims, NC=NC, cids=cids, modifiers=modifiers, **kwargs)

        '''
        CORE
        '''
        if cids is None:
            self.cids = list(range(NC))
        else:
            self.cids = cids
            NC = len(cids)

        self.core = nn.Sequential()
        self.noise_sigma = noise_sigma
        
        conv = layers.ConvLayer(input_dims=input_dims, num_filters=num_subunits[0],
            filter_dims=filter_width[0], NLtype=NLtype,
            output_norm='batch',
            norm_type=norm_type,
            initialize_center=True,
            reg_vals = reg_vals)

        self.core.add_module('layer0', conv)

        for l in range(1, len(filter_width)):
            
            conv = layers.ConvLayer(input_dims=self.core[l-1].output_dims,
                num_filters=num_subunits[l],
                filter_dims=filter_width[l], NLtype=NLtype,
                output_norm='batch',
                norm_type=norm_type,
                initialize_center=True,
                reg_vals = reg_hidden)
            
            self.core.add_module('layer{}'.format(l), conv)

        self.core_subunits = sum(num_subunits)
        self.scaffold = list(range(len(filter_width)))

        '''
        SHIFTER
        '''
        if shifter:
            self.shifter = nn.Sequential(
                nn.Linear(2, 20, bias=False),
                nn.Softplus(),
                nn.Linear(20, 2, bias=True))
        
        if drifter:
            self.drifter = layers.NDNLayer(input_dims=[1, 1, 1, modifiers["offset"][0]],
                            num_filters=2,
                            bias=False,
                            reg_vals = {'d2t': 1e-7})

        '''READOUT'''
        self.readout = layers.ReadoutLayer(input_dims=[self.core_subunits, input_dims[1], input_dims[2], 1],
            num_filters=NC,
            NLtype='lin',
            gauss_type='isotropic',
            reg_vals=reg_readout,
            bias=False,
            )

        self.bias = nn.Parameter(torch.zeros(len(self.cids)))
        self.output_NL = nn.Softplus()


    
    def compute_reg_loss(self):
        rloss = super().compute_reg_loss()
        
        if hasattr(self, 'shifter'):
            rloss += self.shifter(self.reg_placeholder).abs().sum()*10
        
        return rloss
    
    def core_output(self, input):

        # stimulus processing
        out = []
        for layer in self.core:
            input = layer(input)
            if self.training and self.noise_sigma>0:
                input = input + torch.randn_like(input)*self.noise_sigma
            
            out.append(input)

        core_out = torch.cat([out[ind] for ind in self.scaffold], dim=1)

        return core_out

    def forward(self, Xs):
        
        input = Xs['stim']
        core_out = self.core_output(input)

        # shifter
        if hasattr(self, 'shifter'):
            shift = self.shifter(Xs['eyepos'])
            if hasattr(self, 'drifter'):
                shift = self.drifter(Xs[self.offsetstims[0]]) + shift
        else:
            shift = None

        # readout
        output = self.readout(core_out, shift=shift)

        use_offsets = hasattr(self, "offsets")
        use_gains = hasattr(self, "gains")

        offset = self.offval
        if use_offsets:
            for offmod,stim in zip(self.offsets, self.offsetstims):
                offset = offset + offmod(Xs[stim])

        gain = self.gainval
        if use_gains:
            for gainmod,stim in zip(self.gains, self.gainstims):
                gain = gain * (self.gainval + gainmod(Xs[stim]))

        if self.modify:
            output *= gain
            output += offset

        output += self.bias
        # For now assume its just one output, given by the first value of self.ffnet_out
        return self.output_NL(output)


class ShifterDrifter(ShifterModel):

    def __init__(self, input_dims,
            shifter=True,
            drifter=True,
            modifiers=None,
            reg_vals_shifter={'d2x':0.001},
            *args, **kwargs):
        
        super().__init__(input_dims=input_dims,
            shifter=False,
            modifiers=modifiers,
            **kwargs)

        
        if shifter: # time-embedded shifter
            hidden = layers.NDNLayer(input_dims=[1,input_dims[-1],1,2],
                num_filters=20,
                NLtype='softplus',
                norm_type=1,
                bias=False,
                reg_vals = reg_vals_shifter)

            output = layers.NDNLayer(input_dims=hidden.output_dims,
                num_filters=2,
                NLtype='lin',
                norm_type=0,
                bias=True,
                reg_vals = None)
            
            self.shifter = nn.Sequential(hidden, output)

        if drifter:
            self.drifter = layers.NDNLayer(input_dims=[1, 1, 1, modifiers["offset"][0]],
                            num_filters=2,
                            bias=False,
                            reg_vals = {'d2t': 1e-7})
                            
        self.register_buffer('reg_placeholder', torch.zeros((1, 2*input_dims[-1])))
        self.bias = nn.Parameter(torch.zeros(len(self.cids)))
        self.output_NL = nn.Softplus()


    
    def compute_reg_loss(self):
        rloss = super().compute_reg_loss()
        
        if hasattr(self, 'shifter'):
            rloss += self.shifter(self.reg_placeholder).abs().sum()*10
        
        return rloss

    def forward(self, Xs):
        
        input = Xs['stim']
        core_out = self.core_output(input)

        # shifter
        if hasattr(self, 'shifter'):
            shift = self.shifter(Xs['eyepos'])
            if hasattr(self, 'drifter'):
                shift = self.drifter(Xs[self.offsetstims[0]]) + shift
        else:
            shift = None

        # readout
        output = self.readout(core_out, shift=shift)

        use_offsets = hasattr(self, "offsets")
        use_gains = hasattr(self, "gains")

        offset = self.offval
        if use_offsets:
            for offmod,stim in zip(self.offsets, self.offsetstims):
                offset = offset + offmod(Xs[stim])

        gain = self.gainval
        if use_gains:
            for gainmod,stim in zip(self.gains, self.gainstims):
                gain = gain * (self.gainval + gainmod(Xs[stim]))

        if self.modify:
            output *= gain
            output += offset

        output += self.bias
        # For now assume its just one output, given by the first value of self.ffnet_out
        return self.output_NL(output)