'''
This script learns the shifter model for a particular session.
Run this before running preprocess.
'''
#%% Import Libraries
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import torch

from datasets.pixel import Pixel
from datasets.pixel.utils import get_stim_list

%load_ext autoreload
%autoreload 2


#%%
'''
    User-Defined Parameters
'''
SESSION_NAME = '20220610'
spike_sorting = 'kilo'
sesslist = list(get_stim_list().keys())
assert SESSION_NAME in sesslist, "session name %s is not an available session" %SESSION_NAME

datadir = '/mnt/Data/Datasets/MitchellV1FreeViewing/stim_movies/' #'/Data/stim_movies/'
batch_size = 1000
window_size = 35
num_lags = 24
seed = 1234
overwrite = True
retrain = True

#%%
# Process.
train_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset_device = torch.device('cpu')
dtype = torch.float32
sesslist = list(get_stim_list().keys())
NBname = 'shifter_{}'.format(SESSION_NAME)
cwd = os.getcwd()

valid_eye_rad = 6
tdownsample = 2
ds = Pixel(datadir,
    sess_list=[SESSION_NAME],
    requested_stims=['Gabor', 'BackImage'],
    num_lags=num_lags,
    downsample_t=tdownsample,
    download=True,
    valid_eye_rad=valid_eye_rad,
    spike_sorting=spike_sorting,
    fixations_only=False,
    load_shifters=False,
    covariate_requests={
        'fixation_onset': {'tent_ctrs': np.arange(-15, 60, 1)},
        'fixation_num': [],
        'frame_tent': {'ntents': 40}}

    )

print('calculating datafilters')
ds.compute_datafilters(
    to_plot=False,
    verbose=False,
    frac_reject=0.25,
    Lhole=20)

print('[%.2f] fraction of the data used' %ds.covariates['dfs'].mean().item())

gab_inds = np.where(np.in1d(ds.valid_idx, ds.stim_indices[ds.sess_list[0]]['Gabor']['inds']))[0].tolist()

FracDF_include = .2
cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]

crop_window = ds.get_crop_window(inds=gab_inds, win_size=window_size, cids=cids, plot=True)
ds.crop_idx = crop_window

stas = ds.get_stas(inds=gab_inds, square=True)
cids = np.intersect1d(cids, np.where(~np.isnan(stas.std(dim=(0,1,2))))[0])

#%%
# maxsamples = 197144
from NDNT.utils import get_max_samples
if train_device.type == 'cpu':
    maxsamples = len(ds)
else:
    maxsamples = get_max_samples(ds, train_device)

train_inds, val_inds = ds.get_train_indices(max_sample=int(0.85*maxsamples))

train_data = ds[train_inds]
train_data['stim'] = torch.flatten(train_data['stim'], start_dim=1)
val_data = ds[val_inds]
val_data['stim'] = torch.flatten(val_data['stim'], start_dim=1)

cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]
cids = np.intersect1d(cids, np.where(stas.sum(dim=(0,1,2))>0)[0])

input_dims = ds.dims + [ds.num_lags]

#%% Put dataset on GPU if requested


val_device = torch.device('cpu') # if you're cutting it close, put the validation set on the cpu
from foundation.utils.utils import get_datasets, seed_everything
train_dl, val_dl, train_ds, val_ds = get_datasets(train_data, val_data, device=train_device, val_device=val_device, batch_size=batch_size)

#%% 
# from scipy.ndimage import gaussian_filter1d
# %matplotlib inline
# plt.figure(figsize=(10,5))
# plt.plot(gaussian_filter1d(train_ds.covariates['eyepos'][:,1].cpu().numpy(), 100))
# plt.show()
#%%
from foundation.models import CNNdense, Shifter
from copy import deepcopy
from NDNT.training import Trainer, EarlyStopping
from NDNT.utils import load_model


def fit_shifter_model(cp_dir, affine=False, overwrite=False):
    from foundation.utils.utils import memory_clear

    # manually name the model
    name = 'CNN_shifter'
    if affine:
        name = name + '_affine'
    
    # load best model if it already exists
    exists = os.path.isdir(os.path.join(cp_dir, name))
    if exists and not overwrite:
        try:
            smod = load_model(cp_dir, name)

            smod.to(dataset_device)
            val_loss_min = 0
            for data in val_dl:
                out = smod.validation_step(data)
                val_loss_min += out['loss'].item()

            val_loss_min/=len(val_dl)    
            return smod, val_loss_min
        except:
            pass

    os.makedirs(cp_dir, exist_ok=True)

    # parameters of architecture
    num_filters = [20, 20, 20, 20]
    filter_width = [11, 9, 7, 7]
    num_inh = [0]*len(num_filters)
    scaffold = [len(num_filters)-1]

    # build CNN
    cr0 = CNNdense(input_dims,
            num_subunits=num_filters,
            filter_width=filter_width,
            num_inh=num_inh,
            cids=cids,
            bias=False,
            scaffold=scaffold,
            is_temporal=False,
            batch_norm=True,
            window='hamming',
            norm_type=0,
            reg_core=None,
            reg_hidden=None,
            reg_readout={'glocalx':1},
            reg_vals_feat={'l1':0.01, 'norm':1},
                        )
    
    # initialize parameters
    cr0.bias.data = torch.log(torch.exp(ds.covariates['robs'][:,cids].mean(dim=0)) - 1)
    
    # build regularization modules
    cr0.prepare_regularization()

    # wrap in a shifter network
    smod = Shifter(cr0, affine=affine)
    smod.name = name

    optimizer = torch.optim.Adam(smod.parameters(), lr=0.001)
    
    # minimal early stopping patience is all we need here
    earlystopping = EarlyStopping(patience=3, verbose=False)

    trainer = Trainer(optimizer=optimizer,
        device = train_device,
        dirpath = os.path.join(cp_dir, smod.name),
        log_activations=False,
        early_stopping=earlystopping,
        verbose=2,
        max_epochs=100)

    # fit and cleanup memory
    memory_clear()
    trainer.fit(smod, train_dl, val_dl)
    val_loss_min = deepcopy(trainer.val_loss_min)
    del trainer
    memory_clear()
    
    return smod, val_loss_min
    
# %% fit shifter models
from foundation.utils.utils import seed_everything
NBname = f'shifter_{SESSION_NAME}_{seed}'
cwd = os.getcwd()
dirname = os.path.join(cwd, 'data')
cp_dir = os.path.join(dirname, NBname)

# fit shifter with translation only
seed_everything(seed)
mod0, loss0 = fit_shifter_model(cp_dir, affine=False, overwrite=retrain)

# fit shifter with affine
# seed_everything(seed)
# mod1, loss1 = fit_shifter_model(cp_dir, affine=True, overwrite=retrain)

# %%
from foundation.models.utils import plot_stas, eval_model

ll0 = eval_model(mod0, val_dl)

# %%
%matplotlib inline
fig = plt.figure()
plt.hist(ll0)
plt.axvline(0, color='k', linestyle='--')
plt.xlabel("LL (bits/spike)")
plt.ylabel("Count")
plt.show()
# %%
mod0.model.core[0].plot_filters()
# %%
from foundation.datasets.mitchell.pixel.utils import plot_shifter
_,fig00 = plot_shifter(mod0.shifter, show=False)
# %% plot STAs before and after shifting
iix = (train_data['stimid']==0).flatten()
y = (train_data['robs'][iix,:]*train_data['dfs'][iix,:])/train_data['dfs'][iix,:].sum(dim=0).T
stas = (train_data['stim'][iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

_,_,fig02 =  plot_stas(stas.numpy(), title='no shift')

# do shift correction
shift = mod0.shifter(train_data['eyepos'])
stas0 = (mod0.shift_stim(train_data['stim'], shift)[iix,...].T@y).reshape(input_dims[1:] + [-1]).permute(2,0,1,3)

_,_,fig03 =  plot_stas(stas0.detach().numpy(), title='translation')

#%%

cc = 107

nlags = stas.shape[0]//2 # plot half the lags

# zscore STA for specified neuron
sta = stas0[:,:,:,cc] - stas0[:,:,:,cc].mean() 
sta = sta / sta.std()
sta = sta.detach().numpy()

# no shift
sta1 = stas[:,:,:,cc] - stas0[:,:,:,cc].mean() 
sta1 = sta1 / sta1.std()
sta1 = sta1.detach().numpy()

# plot each lag of the STA for the given neuron
fig = plt.figure(figsize=(10,2))
for ilag in range(nlags):
    plt.subplot(2, nlags, ilag+1)
    plt.imshow(sta[ilag,:,:].T, cmap='coolwarm', vmin=-12, vmax=12)
    plt.axis("off")
    plt.subplot(2, nlags, nlags+ilag+1)
    plt.imshow(sta1[ilag,:,:].T, cmap='coolwarm', vmin=-12, vmax=12)
    plt.axis("off")
    
