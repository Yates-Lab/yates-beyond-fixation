

#%% Import Libraries
import sys
import os

import numpy as np

# plotting
import matplotlib.pyplot as plt

# Import torch
import torch
from torch import nn


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

NUM_WORKERS = int(os.cpu_count() / 2)

%load_ext autoreload
%autoreload 2


#%% Load dataset

datadir = '/home/jake/Data/Datasets/Conway/pilot/'
fn = 'data.pt'

data = torch.load(datadir + fn)
data['stim'] = data['stim'][:,0,:,:,:].unsqueeze(1)

#%% crop window
data['stim'] = data['stim'][:,:,10:-10,10:-10,:]
#%% get shifting capabilities

import torch.nn.functional as F

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
        return input.permute(self.permute_order).reshape([-1] + self.new_dims)

    def unfold(self, input):
        return input.reshape([-1] + self.unfold_dims).permute(self.unfold_permute)
    
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
        
        scale = (shift[:,3] + 1.0).clamp(0)

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



class ModelWrapper(nn.Module):
    '''
    Instead of inheriting the Encoder class, wrap models with a class that can be used for training
    '''

    def __init__(self,
            model, # the model to be trained
            loss=nn.PoissonNLLLoss(log_input=False, reduction='mean'), # the loss function to use
            cids = None # which units to use during fitting
            ):
        
        super().__init__()

        if cids is None:
            self.cids = model.cids
        else:
            self.cids = cids
        
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
    


class Shifter(ModelWrapper):
    '''
    Shifter wraps a model with a shifter network
    '''

    def __init__(self, model,
        affine=False, # fit offsets and gains to the shift
        **kwargs):

        super().__init__(model, **kwargs)

        self.affine = affine
        self.input_dims = model.input_dims
        self.name = 'shifter'

        self.fold = Fold2d(model.input_dims)

        self.offsets = nn.Parameter(torch.zeros(1,2))
        self.gains = nn.Parameter(torch.ones(1,2))

        if not affine:
            self.offsets.requires_grad = False
            self.gains.requires_grad = False
        
        # dummy variable to pass in as the eye position for regularization purposes
        self.register_buffer('reg_placeholder', torch.zeros(1,2))
    
    def get_shift(self, eyepos):
        # calculate shift
        return (eyepos[:,[1,0]]/20 - self.offsets) * self.gains
    
    def shift_stim(self, stim, shift):
        '''
        flattened stim as input
        '''
        foldedstim = self.fold(stim.reshape([-1] + self.model.input_dims))        
        return self.fold.unfold(shift_im(foldedstim, shift, False)).flatten(start_dim=1)

    def compute_reg_loss(self):
        
        rloss = self.model.compute_reg_loss()
        rloss += (self.gains - 1).pow(2).sum()
        rloss += self.offsets.pow(2).sum()
        
        return rloss

    def forward(self, batch):
        '''
        The model forward calls the existing model forward after shifting the stimulus
        That's it.
        '''
        
        shift = self.get_shift(batch['eyepos'])

        # replace stimulus in batch with shifted stimulus
        batch['stim'] = self.shift_stim(batch['stim'], shift)
        
        # call model forward
        return self.model(batch['stim'])
    

from torch.utils.data import Dataset

class GenericDataset(Dataset):
    '''
    Generic Dataset can be used to create a quick pytorch dataset from a dictionary of tensors
    
    Inputs:
        Data: Dictionary of tensors. Each key will be a covariate for the dataset.
        device: Device to put each tensor on. Default is cpu.
    '''
    def __init__(self,
        data,
        device=None):

        self.covariates = {}
        for cov in list(data.keys()):
            self.covariates[cov] = data[cov]

        if device is None:
            device = torch.device('cpu')
        
        self.device = device
        
        try:
            if 'stim' in self.covariates.keys() and len(self.covariates['stim'].shape) > 3:
                self.covariates['stim'] = self.covariates['stim'].contiguous(memory_format=torch.channels_last)
        except:
            pass

        self.cov_list = list(self.covariates.keys())
        for cov in self.cov_list:
            self.covariates[cov] = self.covariates[cov].to(self.device)
        
    def __len__(self):

        return self.covariates['stim'].shape[0]

    def __getitem__(self, index):
        return {cov: self.covariates[cov][index,...] for cov in self.cov_list}

#%% get dimensions

input_dims = list(data['stim'].shape[1:])
NC = data['robs'].shape[-1]
print("Input dims: ", input_dims)

#%%

stas = data['stim'].flatten(start_dim=1).T@data['robs']

#%%
c0 = 0
for cc in range(c0, NC):
    plt.figure(figsize=(10,5))
    sta = stas[:,cc].reshape(input_dims)
    for ch in range(input_dims[0]):
        plt.subplot(1,input_dims[0],ch+1)
        plt.imshow(sta[ch,:,:,2].detach().cpu().numpy())
        plt.title(cc)
    plt.show()

#%%
cc = 40# 7
num_lags = input_dims[-1]

plt.figure(figsize=(5,15))
for lag in range(input_dims[-1]):
    sta = stas[:,cc].reshape(input_dims)
    for ch in range(input_dims[0]):
        plt.subplot(num_lags,input_dims[0],lag*input_dims[0]+ch+1)
        plt.imshow(sta[ch,:,:,lag].detach().cpu().numpy(), interpolation='none',
                   vmin=sta[ch,...].min().item(), vmax=sta[ch,...].max().item())

plt.show()


#%%
ds = GenericDataset(data, device=None)
ds.covariates['stim'] = ds.covariates['stim'].flatten(start_dim=1)


# get dataloader
from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=1000, shuffle=False, num_workers=os.cpu_count()//2)

#%%
fold = Fold2d(input_dims)
stas1 = 0
from tqdm import tqdm
for batch in tqdm(dl):
    stim = shift_im(fold(batch['stim'].reshape([-1]+input_dims)), batch['eyepos'][:,[1,0]]/input_dims[1], False, mode='nearest')
    stim = fold.unfold(stim)
    stas1 += (stim.flatten(start_dim=1).T@batch['robs']).detach().cpu()

#%%
c0 = 0
for cc in range(c0, NC):
    plt.figure(figsize=(10,5))
    sta = stas1[:,cc].reshape(input_dims)
    for ch in range(input_dims[0]):
        plt.subplot(1,input_dims[0],ch+1)
        plt.imshow(sta[ch,:,:,2].numpy())
    plt.title(cc)
    plt.show()

#%%
cc = 7 #cc + 1 #7
num_lags = input_dims[-1]

plt.figure(figsize=(5,15))
sta0 = stas[:,cc].reshape(input_dims)
sta = stas1[:,cc].reshape(input_dims)
vmax = sta[ch,...].max().item()
vmin = sta[ch,...].min().item()

for lag in range(input_dims[-1]):
    
    for ch in range(input_dims[0]):
        plt.subplot(num_lags,input_dims[0]*2,lag*input_dims[0]*2+ch+1)
        plt.imshow(sta0[ch,:,:,lag].detach().cpu().numpy(), interpolation='none',
                   vmin=vmin, vmax=vmax)
    
    for ch in range(input_dims[0]):
        plt.subplot(num_lags,input_dims[0]*2,lag*input_dims[0]*2+input_dims[0]+ch+1)
        plt.imshow(sta[ch,:,:,lag].detach().cpu().numpy(), interpolation='none',
                   vmin=vmin, vmax=vmax)

plt.show()

#%%
sys.path.insert(0, '/home/jake/Data/Repos')
from NDNT.modules.layers import NDNLayer
from NDNT.training import Trainer

class LNLN(nn.Module):

    def __init__(self, input_dims,
                 num_hidden, 
                 NC,
                 core_reg_vals,
                 readout_reg_vals,
                 NLtype='softplus', bias=True):
        
        super().__init__()

        self.input_dims = input_dims
        self.num_hidden = num_hidden
        self.core = NDNLayer(input_dims=input_dims, num_filters=num_hidden, reg_vals=core_reg_vals, NLtype=NLtype, bias=False, output_norm='batch')
        self.readout = NDNLayer(input_dims=[num_hidden, 1, 1, 1], num_filters=NC, reg_vals=readout_reg_vals, NLtype='softplus', bias=bias)

    def forward(self, x):
        return self.readout(self.core(x))

    def prepare_regularization(self, normalize_reg=False):
        
        self.core.reg.normalize = normalize_reg
        self.core.reg.build_reg_modules()

        self.readout.reg.normalize = normalize_reg
        self.readout.reg.build_reg_modules()
        
    def compute_reg_loss(self):
        
        reg = self.core.compute_reg_loss()
        reg += self.readout.compute_reg_loss()
        return reg

lnln = LNLN(input_dims=input_dims, num_hidden=NC//2, NC=NC, core_reg_vals={'norm2':1,  'glocalx': 1}, readout_reg_vals={'l1': 0.0001}, NLtype='softplus', bias=True)
model = Shifter(lnln, affine=True, cids=list(range(NC)))
model.prepare_regularization()

import gc
torch.cuda.empty_cache()
gc.collect()


# %%
# split into train and test
train_ds, test_ds = torch.utils.data.random_split(ds, [int(len(ds)*.8), len(ds)-int(len(ds)*.8)])
train_dl = DataLoader(train_ds, batch_size=1000, shuffle=True, num_workers=os.cpu_count()//2)
test_dl = DataLoader(test_ds, batch_size=1000, shuffle=True, num_workers=os.cpu_count()//2)

batch = next(iter(train_dl))
yhat = model(batch)

# %%

lnln = LNLN(input_dims=input_dims, num_hidden=NC//2, NC=NC, core_reg_vals={'norm2':1,  'glocalx': 10, 'd2t': 1}, readout_reg_vals={'l1': 0.0001}, NLtype='softplus', bias=True)
model = Shifter(lnln, affine=True, cids=list(range(NC)))
model.prepare_regularization()

from NDNT.training import Trainer, EarlyStopping

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

earlystopping = EarlyStopping(patience=10, verbose=False)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
trainer = Trainer(optimizer=optimizer,
            device = device,
            max_epochs=1000,
            dirpath=os.path.join('checkpoints', 'shift'),
            verbose=2,
            scheduler=scheduler,
            scheduler_after='epoch',
            scheduler_metric='val_loss',
            early_stopping=earlystopping)

trainer.fit(model, train_dl, test_dl)
# %%

model.model.core.plot_filters()
# %%
trainer.fit(model, train_dl, test_dl)
# %%
