import torch
import torch.nn as nn
from NDNT.modules import layers as layers

class DenseReadout(nn.Module):

    def __init__(self,
        input_dims, 
        num_filters,
        reg_vals=None,
        reg_vals_feat={'l1':0.01, 'l2':0.01},
        **kwargs):

        super().__init__()
        
        assert len(input_dims)==4, 'DenseReadout: input_dims must have form [channels, height, width, 1]'
        assert input_dims[-1]==1, 'DenseReadout cannot have time lags'

        self.input_dims=input_dims
        self.num_filters=num_filters
        in_channels = int(input_dims[0])
        spat_dims = [1] + input_dims[1:] 
        self.feature = layers.NDNLayer(input_dims=[in_channels, 1, 1, 1], num_filters=num_filters, reg_vals=reg_vals_feat, **kwargs)
        self.space = layers.NDNLayer(input_dims=spat_dims, num_filters=num_filters, reg_vals=reg_vals, **kwargs)
    
    def forward(self, x):
        
        # get weights
        wf = self.feature.preprocess_weights()
        ws = self.space.preprocess_weights()
        ws = ws.view(self.space.filter_dims[1:3] + [self.space.num_filters])
        # process input dims
        x = x.view([-1]+self.input_dims)

        return torch.einsum('bcxyl,xyn,cn->bn',x,ws,wf)

    def build_reg_modules(self, normalize_reg=False):
        
        self.space.reg.normalize = normalize_reg
        self.feature.reg.normalize = normalize_reg
        self.space.reg.build_reg_modules()
        self.feature.reg.build_reg_modules()

    def compute_reg_loss(self):
        
        rloss = self.feature.compute_reg_loss()
        rloss += self.space.compute_reg_loss()

        return rloss

def buildConvCore(
    input_dims,
    num_subunits,
    filter_width,
    norm_type,
    num_inh,
    scaffold,
    NLtype,
    batch_norm,
    is_temporal,
    reg_core,
    reg_hidden,
    window,
    padding,
    bias,
    pos_constraint_hard=False):

    nlayers = len(num_subunits)
    core = nn.Sequential()
    if is_temporal:
        layer = layers.TconvLayer(input_dims=input_dims,
            num_filters=num_subunits[0],
            filter_dims=filter_width[0],
            norm_type=norm_type,
            num_inh=num_inh[0],
            initialize_center=False,
            NLtype=NLtype,
            padding='spatial',
            output_norm='batch' if batch_norm else None,
            reg_vals = reg_core)
    else:
        layer = layers.ConvLayer(input_dims=input_dims,
            num_filters=num_subunits[0],
            filter_dims=filter_width[0],
            norm_type=norm_type,
            num_inh=num_inh[0],
            window=window,
            padding=padding,
            initialize_center=False,
            NLtype=NLtype,
            output_norm='batch' if batch_norm else None,
            reg_vals = reg_core)
        
    core.add_module('layer0', layer)
    core_subunits = 0
    if 0 in scaffold:
        core_subunits += num_subunits[0]

    for l in range(1,nlayers):
        if num_inh[l-1] > 0 and pos_constraint_hard:
            pos_constraint = True
        else:
            pos_constraint = False

        if l==1 and is_temporal:
            reg_list = reg_core
        else:
            reg_list = reg_hidden

        layer = layers.ConvLayer(input_dims=core[l-1].output_dims,
            num_filters=num_subunits[l],
            filter_dims=filter_width[l],
            norm_type=norm_type,
            num_inh=num_inh[l],
            initialize_center=True,
            NLtype=NLtype,
            window=window,
            padding=padding,
            pos_constraint=pos_constraint,
            output_norm='batch' if batch_norm else None,
            bias=bias,
            reg_vals = reg_list)

        core.add_module('layer{}'.format(l), layer)
        if l in scaffold:
            core_subunits += num_subunits[l]

    return core, core_subunits
class CNNdense(nn.Module):

    def __init__(self,
        input_dims,
        num_subunits,
        filter_width,
        scaffold=None,
        num_inh=None,
        is_temporal=False,
        NLtype='relu',
        batch_norm=False,        
        norm_type=0,
        padding='same',
        bias=True,
        window=None,
        reg_core={'d2x':0.001, 'd2t':0.001, 'center':0.1},
        reg_hidden={'l2':0.01},
        reg_readout={'l2':0.01},
        reg_vals_feat={'l1':0.001},
        cids=None,
        **kwargs):

        super().__init__()

        self.cids = cids
        self.name = 'CnnDense'
        
        self.input_dims = input_dims

        nlayers = len(num_subunits)
        if num_inh is None:
            num_inh = [0]*nlayers
        
        if scaffold is None:
            self.scaffold = list(range(nlayers))
        else:
            self.scaffold = scaffold

        '''
        build core. TODO: this should probably become its own object
        '''
        self.core, self.core_subunits = buildConvCore(input_dims,
            num_subunits,
            filter_width,
            norm_type,
            num_inh,
            self.scaffold,
            NLtype,
            batch_norm,
            is_temporal,
            reg_core,
            reg_hidden,
            window,
            padding,
            bias)
        
    #     self.core = nn.Sequential()
    #     self.name = 'CNN'
    #     self.NC = len(self.cids)

    # if is_temporal:
    #     layer = layers.TconvLayer(input_dims=input_dims,
    #         num_filters=num_subunits[0],
    #         filter_dims=filter_width[0],
    #         norm_type=norm_type,
    #         num_inh=num_inh[0],
    #         initialize_center=False,
    #         NLtype=NLtype,
    #         padding='spatial',
    #         output_norm='batch' if batch_norm else None,
    #         reg_vals = reg_core)
    # else:
    #     layer = layers.ConvLayer(input_dims=input_dims,
    #         num_filters=num_subunits[0],
    #         filter_dims=filter_width[0],
    #         norm_type=norm_type,
    #         num_inh=num_inh[0],
    #         window=window,
    #             padding=padding,
    #         initialize_center=False,
    #         NLtype=NLtype,
    #         output_norm='batch' if batch_norm else None,
    #         reg_vals = reg_core)
        
    #     self.core.add_module('layer0', layer)
    #     self.core_subunits = 0
    #     if 0 in self.scaffold:
    #         self.core_subunits += num_subunits[0]

    # for l in range(1,nlayers):
    #     if num_inh[l-1] > 0:
    #         pos_constraint = True
    #     else:
    #         pos_constraint = False

    #     if l==1 and is_temporal:
    #         reg_list = reg_core
    #     else:
    #         reg_list = reg_hidden

    #         layer = layers.ConvLayer(input_dims=self.core[l-1].output_dims,
    #         num_filters=num_subunits[l],
    #         filter_dims=filter_width[l],
    #         norm_type=norm_type,
    #         num_inh=num_inh[l],
    #         initialize_center=True,
    #         NLtype=NLtype,
    #         window=window,
    #             padding=padding,
    #         pos_constraint=pos_constraint,
    #         output_norm='batch' if batch_norm else None,
    #         bias=bias,
    #         reg_vals = reg_list)

    #         self.core.add_module('layer{}'.format(l), layer)
    #         if l in self.scaffold:
    #             self.core_subunits += num_subunits[l]

        self.nlayers = nlayers
        self.readout = DenseReadout(input_dims=[self.core_subunits, self.core[-1].output_dims[1], self.core[-1].output_dims[2], 1],
            num_filters=len(self.cids),
            pos_constraint = self.core[-1].pos_constraint,
            NLtype='lin',
            bias=False,
            reg_vals = reg_readout,
            reg_vals_feat = reg_vals_feat,
            )

        self.bias = nn.Parameter(torch.zeros(len(self.cids)))
        self.output_NL = nn.Softplus()
    
    def prepare_regularization(self, normalize_reg = False):
        
        for layer in self.core:
            layer.reg.normalize = normalize_reg
            layer.reg.build_reg_modules()

        if hasattr(self, 'readout'):
            self.readout.build_reg_modules(normalize_reg=normalize_reg)
    
    def compute_reg_loss(self):
        rloss = 0
        for layer in self.core:
            rloss += layer.compute_reg_loss()
        
        if hasattr(self, 'readout'):
            if hasattr(self.readout, '__iter__'):
                for layer in self.readout:
                    rloss += layer.compute_reg_loss()
            else:
                rloss += self.readout.compute_reg_loss()
            
        return rloss


    def core_output(self, input):

        out = []
        for l,layer in enumerate(self.core):
            input = layer(input)
            if l in self.scaffold:
                out.append(input.reshape([-1] + layer.output_dims)[...,-1])

        core_out = torch.cat(out, dim=1)
        return core_out

    def pre_epoch(self):
        for layer in self.core:
            if hasattr(layer, "pre_epoch"):
                layer.pre_epoch()

    def forward(self, Xs):
        
        

        # stimulus processing
        input = Xs['stim']
        core_out = self.core_output(input)

        # readout
        output = self.readout(core_out)

        output += self.bias

        # For now assume its just one output, given by the first value of self.ffnet_out
        return self.output_NL(output)


class CNN(nn.Module):

    def __init__(self,
        input_dims,
        num_subunits,
        filter_width,
        scaffold=None,
        num_inh=None,
        is_temporal=False,
        NLtype='relu',
        batch_norm=False,        
        norm_type=0,
        NC=None,
        bias=True,
        reg_core={'d2x':0.001, 'd2t':0.001, 'center':0.1},
        reg_hidden={'l2':0.01},
        reg_readout={'l2':0.01},
        cids=None,
        **kwargs):

        super().__init__(input_dims,NC=NC,cids=cids,**kwargs)

        nlayers = len(num_subunits)
        if num_inh is None:
            num_inh = [0]*nlayers
        
        if scaffold is None:
            self.scaffold = list(range(nlayers))
        else:
            self.scaffold = scaffold

        self.core = nn.Sequential()
        self.name = 'CNN'
        self.NC = len(self.cids)

        if is_temporal:
            layer = layers.TconvLayer(input_dims=input_dims,
                num_filters=num_subunits[0],
                filter_dims=filter_width[0],
                norm_type=norm_type,
                num_inh=num_inh[0],
                initialize_center=False,
                NLtype=NLtype,
                padding='spatial',
                output_norm='batch' if batch_norm else None,
                reg_vals = reg_core)
        else:
            layer = layers.ConvLayer(input_dims=input_dims,
                num_filters=num_subunits[0],
                filter_dims=filter_width[0],
                norm_type=norm_type,
                num_inh=num_inh[0],
                initialize_center=False,
                NLtype=NLtype,
                output_norm='batch' if batch_norm else None,
                reg_vals = reg_core)
            
        self.core.add_module('layer0', layer)
        self.core_subunits = 0
        if 0 in self.scaffold:
            self.core_subunits += num_subunits[0]

        for l in range(1,nlayers):
            if num_inh[l-1] > 0:
                pos_constraint = True
            else:
                pos_constraint = False

            if l==1 and is_temporal:
                reg_list = reg_core
            else:
                reg_list = reg_hidden

            layer = layers.ConvLayer(input_dims=self.core[l-1].output_dims,
                num_filters=num_subunits[l],
                filter_dims=filter_width[l],
                norm_type=norm_type,
                num_inh=num_inh[l],
                initialize_center=True,
                NLtype=NLtype,
                pos_constraint=pos_constraint,
                output_norm='batch' if batch_norm else None,
                bias=bias,
                reg_vals = reg_list)

            self.core.add_module('layer{}'.format(l), layer)
            if l in self.scaffold:
                self.core_subunits += num_subunits[l]

        self.nlayers = nlayers
        self.readout = layers.ReadoutLayer(input_dims=[self.core_subunits, input_dims[1], input_dims[2], 1],
            num_filters=len(self.cids),
            pos_constraint = self.core[-1].pos_constraint,
            NLtype='lin',
            gauss_type='isotropic',
            bias=False,
            reg_vals = reg_readout
            )

        self.bias = nn.Parameter(torch.zeros(len(self.cids)))
        self.output_NL = nn.Softplus()


    def core_output(self, input):

        out = []
        for l,layer in enumerate(self.core):
            input = layer(input)
            if l in self.scaffold:
                out.append(input.reshape([-1] + layer.output_dims)[...,-1])

        core_out = torch.cat(out, dim=1)
        return core_out

    def forward(self, Xs):
        
        

        # stimulus processing
        input = Xs['stim']
        core_out = self.core_output(input)

        if self.modify:
            use_offsets = hasattr(self, "offsets")
            use_gains = hasattr(self, "gains")

            modify_core = "core" in self.modifierstage

            # modify core if requested
            if use_offsets and modify_core:
                offset = self.offval

                for offmod,stim,stage in zip(self.offsets, self.offsetstims, self.modifierstage):
                    if stage=='core':
                        offset = offset + offmod(Xs[stim])
            
            if use_gains and modify_core:
                gain = self.gainval
                for gainmod,stim,stage in zip(self.gains, self.gainstims, self.modifierstage):
                    if stage=='core':
                        gain = gain * (self.gainval + gainmod(Xs[stim]))

            if self.modify and modify_core:
                core_out *= gain
                core_out += offset

        # readout
        output = self.readout(core_out)

        if self.modify:
            modify_readout = "readout" in self.modifierstage

            # modify readout if requested
            if use_offsets and modify_readout:
                offset = self.offval

                for offmod,stim,stage in zip(self.offsets, self.offsetstims, self.modifierstage):
                    if stage=='readout':
                        offset = offset + offmod(Xs[stim])
            
            if use_gains and modify_readout:
                gain = self.gainval
                for gainmod,stim,stage in zip(self.gains, self.gainstims, self.modifierstage):
                    if stage=='readout':
                        gain = gain * (self.gainval + gainmod(Xs[stim]))

            if self.modify and modify_readout:
                output *= gain
                output += offset
        

        output += self.bias

        # For now assume its just one output, given by the first value of self.ffnet_out
        return self.output_NL(output)