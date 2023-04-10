import torch
import torch.nn as nn
from NDNT import PoissonLoss_datafilter

class ModelWrapper(nn.Module):
    '''
    Instead of inheriting the Encoder class, wrap models with a class that can be used for training
    '''

    def __init__(self,
            model, # the model to be trained
            loss=PoissonLoss_datafilter(), # the loss function to use
            cids = None # which units to use during fitting
            ):
        
        super().__init__()

        if cids is None:
            self.cids = model.cids
        
        self.model = model
        if hasattr(model, 'name'):
            self.name = model.name
        else:
            self.name = 'unnamed'

        self.loss = loss

    
    def compute_reg_loss(self):
        
        return self.model.compute_reg_loss()

    def prepare_regularization(self, normalize_reg = False):
        
        self.model.prepare_regularization(normalize_reg=normalize_reg)
    
    def forward(self, batch):

        return self.model(batch)

    def training_step(self, batch, batch_idx=None):  # batch_indx not used, right?
        
        y = batch['robs'][:,self.cids]
        y_hat = self(batch)

        if 'dfs' in batch.keys():
            dfs = batch['dfs'][:,self.cids]
            loss = self.loss(y_hat, y, dfs)
        else:
            loss = self.loss(y_hat, y)

        regularizers = self.compute_reg_loss()

        return {'loss': loss + regularizers, 'train_loss': loss, 'reg_loss': regularizers}

    def validation_step(self, batch, batch_idx=None):
        
        y = batch['robs'][:,self.cids]
        
        y_hat = self(batch)

        if 'dfs' in batch.keys():
            dfs = batch['dfs'][:,self.cids]
            loss = self.loss(y_hat, y, dfs)
        else:
            loss = self.loss(y_hat, y)

        return {'loss': loss, 'val_loss': loss, 'reg_loss': None}
    

'''
We use Fold2d to take the raw stimulus dimensions [Batch x Channels x Height x Width x Lags]
and "fold" the lags into the channel dimension so we can do 2D convolutions on time-embedded stimuli
'''
class Fold2d(nn.Module):
    __doc__ = r"""Folds the lags dimension of a 4D tensor into the channel dimension so that 2D convolutions can be applied to the spatial dimensions.
    In the simplest case, the output value of the layer with input size
    :math:`(N, C, H, W, T)` is :math:`(N, C\timesT, H, W)`
    
    the method unfold will take folded input of size :math:`(N, C\timesT, H, W)` and output the original dimensions :math:`(N, C, H, W, T)`
    """

    def __init__(self, dims=None):

        self.orig_dims = dims
        super(Fold2d, self).__init__()
        
        self.permute_order = (0,1,4,2,3)
        self.new_dims = [dims[3]*dims[0], dims[1], dims[2]]
        self.unfold_dims = [dims[0], dims[3], dims[1], dims[2]]
        self.unfold_permute = (0, 1, 3, 4, 2)
    
    def forward(self, input):

        return self.fold(input)
    
    def fold(self, input):
        return input.permute(self.permute_order).view([-1] + self.new_dims)

    def unfold(self, input):
        return input.reshape([-1] + self.unfold_dims).permute(self.unfold_permute)