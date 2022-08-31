import torch
import torch.nn as nn
from NDNT.metrics.poisson_loss import PoissonLoss_datafilter
from NDNT.modules import layers as layers

class Encoder(nn.Module):

    def __init__(self, input_dims,
            NC=None,
            cids=None,
            modifiers=None,
            *args, **kwargs):
        
        super().__init__()

        if cids is None:
            self.cids = list(range(NC))
        else:
            self.cids = cids
            NC = len(cids)
        
        self.input_dims = input_dims
        self.name = 'base'

        self.loss = PoissonLoss_datafilter()
        self.loss.unit_normalization = False

        '''
        MODIFIERS
        '''
        self.modify = False
        

        if modifiers is not None:

            # initialize variables for modifier: these all need to be here regardless of whether the modifiers are used so we can load the model checkpoints
            self.offsets = nn.ModuleList()
            self.gains = nn.ModuleList()
            self.offsetstims = []
            self.gainstims = []
            self.register_buffer("offval", torch.zeros(1))
            self.register_buffer("gainval", torch.ones(1))

            """
            modifier is a hacky addition to the model to allow for offsets and gains at a certain stage in the model
            The default stage is after the readout
            example modifier input:
            modifier = {'stimlist': ['frametent', 'saccadeonset'],
            'gain': [40, None],
            'offset':[40,20],
            'stage': "readout",
            'outdims: gd.NC}
            """
            if type(modifiers)==dict:
                self.modify = True

                nmods = len(modifiers['stimlist'])
                assert nmods==len(modifiers["offset"]), "Encoder: modifier specified incorrectly"
                
                if 'stage' not in modifiers.keys():
                    modifiers['stage'] = ["readout"]*nmods
                
                
                self.modifierstage = []
                for imod in range(nmods):
                    if modifiers["offset"][imod] is not None:
                        self.offsetstims.append(modifiers['stimlist'][imod])

                        # set the output dims (this hast to match either the readout output the whole core is modulated)
                        if modifiers['stage'][imod] =="readout":
                            outdims = NC
                        elif modifiers['stage'][imod] =="core":
                            outdims = 1
                        
                        if 'outdims' in modifiers.keys():
                            outdims = modifiers['outdims'][imod]
                            
                        self.modifierstage.append(modifiers["stage"][imod])

                        self.offsets.append(
                            layers.NDNLayer(input_dims=[1, 1, 1, modifiers["offset"][imod]],
                            num_filters=outdims,
                            bias=False,
                            reg_vals = {'d2t': 1e-7})
                        )
                    if modifiers["gain"][imod] is not None:
                        self.gainstims.append(modifiers['stimlist'][imod])
                        
                        self.gains.append(
                            layers.NDNLayer(input_dims=[1,1, 1, modifiers["gain"][imod]],
                            num_filters=outdims,
                            bias=False,
                            reg_vals = {'d2t': 1e-7})
                        )
        else:
            self.modify = False

        self.register_buffer('reg_placeholder', torch.zeros(1,2))
        self.var_pen = 0
        self.output_NL = nn.Softplus()
    
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
        
        if hasattr(self, 'offsets'):
            for layer in self.offsets:
                rloss += layer.compute_reg_loss()
        
        if hasattr(self, 'gains'):
            for layer in self.gains:
                rloss += layer.compute_reg_loss()
            
        return rloss

    def prepare_fit(self, train_dl):

        if self.loss.unit_normalization:
            
            r0 = 0
            dfs = 0
            for data in train_dl:
                r0 += data['robs'][:,self.cids].sum(0)
                dfs += data['dfs'][:,self.cids].sum(0)
            
            self.loss.set_unit_normalization(r0/dfs)


    def prepare_regularization(self, normalize_reg = False):
        
        for layer in self.core:
            layer.reg.normalize = normalize_reg
            layer.reg.build_reg_modules()

        if hasattr(self, 'readout'):
            if hasattr(self.readout, '__iter__'):
                for layer in self.readout:
                    layer.reg.normalize = normalize_reg
                    layer.reg.build_reg_modules()
            else:
                self.readout.reg.normalize = normalize_reg
                self.readout.reg.build_reg_modules()
        
        if hasattr(self, 'offsets'):
            for layer in self.offsets:
                layer.reg.normalize = normalize_reg
                layer.reg.build_reg_modules()
        
        if hasattr(self, 'gains'):
            for layer in self.gains:
                layer.reg.normalize = normalize_reg
                layer.reg.build_reg_modules()
    

    def training_step(self, batch, batch_idx=None):  # batch_indx not used, right?
        
        y = batch['robs'][:,self.cids]
        dfs = batch['dfs'][:,self.cids]

        y_hat = self(batch)

        loss = self.loss(y_hat, y, dfs)

        regularizers = self.compute_reg_loss()

        regularizers = regularizers - self.var_pen*torch.sigmoid(y_hat.var(dim=0).sum())

        return {'loss': loss + regularizers, 'train_loss': loss, 'reg_loss': regularizers}

    def validation_step(self, batch, batch_idx=None):
        
        y = batch['robs'][:,self.cids]
        dfs = batch['dfs'][:,self.cids]

        y_hat = self(batch)

        loss = self.loss(y_hat, y, dfs)
        
        return {'loss': loss, 'val_loss': loss, 'reg_loss': None}