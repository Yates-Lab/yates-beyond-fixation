import torch
import torch.nn as nn

from NDNT.modules import layers as layers
from .base import ModelWrapper, Fold2d

import torch
import torch.nn as nn
import torch.nn.functional as F


def shift_im(stim, shift, affine=False, mode='bilinear'):
    '''
    Primary function for shifting the intput stimulus
    Inputs:
        stim [Batch x channels x height x width] (use Fold2d to fold lags if necessary)
        shift [Batch x 2] or [Batch x 4] if translation only or affine
        affine [Boolean] set to True if using affine transformation
        mode [str] 'bilinear' (default) or 'nearest'
        NOTE: mode must be bilinear during fitting otherwise the gradients don't propogate well
    '''

    batch_size = stim.shape[0]

    # build affine transformation matrix size = [batch x 2 x 3]
    affine_trans = torch.zeros((batch_size, 2, 3) , dtype=stim.dtype, device=stim.device)
    
    if affine:
        # fill in rotation and scaling
        costheta = torch.cos(shift[:,2].clamp(-.5, .5))
        sintheta = torch.sin(shift[:,2].clamp(-.5, .5))
        
        scale = shift[:,3]**2 + 1.0

        affine_trans[:,0,0] = costheta*scale
        affine_trans[:,0,1] = -sintheta*scale
        affine_trans[:,1,0] = sintheta*scale
        affine_trans[:,1,1] = costheta*scale

    else:
        # no rotation or scaling
        affine_trans[:,0,0] = 1
        affine_trans[:,0,1] = 0
        affine_trans[:,1,0] = 0
        affine_trans[:,1,1] = 1

    # translation
    affine_trans[:,0,2] = shift[:,0]
    affine_trans[:,1,2] = shift[:,1]

    grid = F.affine_grid(affine_trans, stim.shape, align_corners=False)

    return F.grid_sample(stim, grid, mode=mode, align_corners=False)

class Shifter(ModelWrapper):
    '''
    Shifter wraps a model with a shifter network
    '''

    def __init__(self, model,
        num_hidden=20,
        affine=False,
        **kwargs):

        super().__init__(model, **kwargs)

        self.affine = affine
        self.input_dims = model.input_dims
        self.name = model.name + '_shifter'
        self.cids = model.cids

        self.fold = Fold2d(model.input_dims)
        if affine:
            self.shifter = nn.Sequential(
                nn.Linear(2, num_hidden, bias=False),
                nn.Softplus(), 
                nn.Linear(num_hidden, 4, bias=True))
        else:
            self.shifter = nn.Sequential(
                nn.Linear(2, 20, bias=False),
                nn.Softplus(),
                nn.Linear(20, 2, bias=True))
        
        # dummy variable to pass in as the eye position for regularization purposes
        self.register_buffer('reg_placeholder', torch.zeros(1,2))
    
    def shift_stim(self, stim, shift):
        '''
        flattened stim as input
        '''
        foldedstim = self.fold(stim.view([-1] + self.model.input_dims))        
        return self.fold.unfold(shift_im(foldedstim, shift, self.affine)).flatten(start_dim=1)

    def compute_reg_loss(self):
        
        rloss = self.model.compute_reg_loss()
        
        rloss += self.shifter(self.reg_placeholder).abs().sum()*10
        
        return rloss

    def forward(self, batch):
        '''
        The model forward calls the existing model forward after shifting the stimulus
        That's it.
        '''
        # calculate shift
        shift = self.shifter(batch['eyepos'])

        # replace stimulus in batch with shifted stimulus
        batch['stim'] = self.shift_stim(batch['stim'], shift)
        # call model forward
        return self.model(batch)