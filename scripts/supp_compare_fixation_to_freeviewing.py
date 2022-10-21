
#%%
import sys
import os

# setup paths
sys.path.insert(0, '/home/jake/Data/Repos/')
sys.path.insert(0, '/home/jake/Data/Repos/yates-beyond-fixation/scripts/')
fig_dir = '/home/jake/Data/Repos/yates-beyond-fixation/figures/supp_fix_freeview'

from NDNT.utils.NDNutils import ensure_dir
ensure_dir(fig_dir)

sys.path.append("../")
# pytorch
import numpy as np

# plotting
import matplotlib.pyplot as plt

# Import torch
import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

NUM_WORKERS = int(os.cpu_count() / 2)

%load_ext autoreload
%autoreload 2
%matplotlib inline

#%% Load fixc dataset first to see if this is even viable
from datasets.pixel import Pixel
from models.utils import plot_stas

sessname = '20220610'
datadir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'

#%% Load dataset

valid_eye_rad = 6
ds = Pixel(datadir,
    sess_list=[sessname],
    requested_stims=['Gabor'],
    num_lags=12,
    downsample_t=2,
    download=True,
    valid_eye_rad=valid_eye_rad,
    fixations_only=True,
    load_shifters=True,
    spike_sorting='kilo',
    )

# print("Calculating Datafilters...")
# ds.compute_datafilters(to_plot=True)
print("Done")

from copy import deepcopy
dims_orig = deepcopy(ds.dims)

# load fixation datasets
dsfix = Pixel(datadir,
    sess_list=[sessname + 'fix'],
    requested_stims=['FixFlashGabor'],
    num_lags=12,
    downsample_t=2,
    download=True,
    valid_eye_rad=8,
    ctr=np.array([0,0]),
    fixations_only=True,
    load_shifters=False,
    spike_sorting='kilo',
    )


dsfixc = Pixel(datadir,
    sess_list=[sessname + 'fixc'],
    requested_stims=['FixFlashGabor'],
    num_lags=12,
    downsample_t=2,
    download=True,
    valid_eye_rad=valid_eye_rad,
    ctr=np.array([0,0]),
    fixations_only=True,
    load_shifters=False,
    spike_sorting='kilo',
    )

#%% get cell inclusion
FracDF_include = .2
cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]
ppd = ds.stim_indices[ds.sess_list[0]][ds.requested_stims[0]]['ppd']
#%% get STAs at full stimulus resolution
'''
We need to get the full stimulus resolution STAs for each cell under fixation and free-viewing
'''

# get STA on the free-viewing Gaborium stimulus
gab_inds = np.where(np.in1d(ds.valid_idx, ds.stim_indices[ds.sess_list[0]]['Gabor']['inds']))[0].tolist()
stas_full = ds.get_stas(inds=gab_inds, square=False)

mu, bestlag = plot_stas(stas_full.detach().numpy())

# get the STA during fixation

# corrected with the measured gaze position
stas_fixc_full = dsfixc.get_stas(square=False)

# get the "raw" fixation STAs in a restricted radius.
rgx0 = 0
rgy0 = 0
rad = .5 # degrees from center

ctr_inds = np.where(np.hypot(dsfix.covariates['eyepos'][dsfix.valid_idx,0].cpu().numpy() - rgx0, dsfix.covariates['eyepos'][dsfix.valid_idx,1].cpu().numpy() - rgy0) < rad)[0].tolist()
stas_fix_full = dsfix.get_stas(inds=ctr_inds, square=False)

#%% Find spatial window to crop so the dataset will fit in GPU memory

win_size = 35 # size of crop window in pixels

ds.crop_idx = [0, dims_orig[1], 0, dims_orig[2]] # set original crop index


sta2 = ds.get_stas(inds=gab_inds, square=True)
spower = sta2[...,cids].std(dim=0)
spatial_power = torch.einsum('whn,n->wh', spower, ds.covariates['robs'][:,cids].sum(dim=0)/ds.covariates['robs'][:,cids].sum())
spatial_power[spatial_power < .5*spatial_power.max()] = 0 # remove noise floor
spatial_power /= spatial_power.sum()

xx,yy = torch.meshgrid(torch.arange(0, sta2.shape[1]), torch.arange(0, sta2.shape[2]))

ctr_x = (spatial_power * yy).sum().item()
ctr_y = (spatial_power * xx).sum().item()

plt.imshow(spatial_power.detach().numpy())
plt.plot(ctr_x, ctr_y, 'rx')

x0 = int(ctr_x) - int(np.ceil(win_size/2))
x1 = int(ctr_x) + int(np.floor(win_size/2))
y0 = int(ctr_y) - int(np.ceil(win_size/2))
y1 = int(ctr_y) + int(np.floor(win_size/2))

plt.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], 'r')
#%% redo stas with cropped window
ds.crop_idx = [y0,y1,x0,x1]
dsfixc.crop_idx = [y0,y1,x0,x1]
dsfix.crop_idx = [y0,y1,x0,x1]

#%%

# stas_all = ds.get_stas()

# stas_all2 = ds.get_stas(square=True)
# stas_all2 = stas_all2.detach().numpy()

# #%%
# _ = plot_stas(stas_all.detach().numpy())

#%% get units with significant STAs


data = ds[gab_inds]
mu = data['stim'].mean(dim=(0,1))
sd = data['stim'].std(dim=(0,1))

thresh = 0.001 # significance at the p < thresh level
robs = data['robs']*data['dfs']
ny = robs.sum(dim=0)
stas_ = torch.einsum('ncxyt,nm->xytm', data['stim'], data['robs'])
stas_ /= ny[None,None,None,:]
stas_ = stas_.permute(2,0,1,3).detach().cpu().numpy()

stas_2 = torch.einsum('ncxyt,nm->xytm', data['stim']**2, data['robs'])
stas_2 /= ny[None,None,None,:]
stas_2 = stas_2.permute(2,0,1,3).detach().cpu().numpy()

import scipy.stats as st
snum = st.norm.isf(thresh/ds.num_lags/ds.NC)
ci = mu[:,:,:,None] + snum*sd[:,:,:,None]/ny.sqrt()
ci = ci.permute(2,0,1,3).detach().cpu().numpy()

sig = np.mean(np.abs(stas_)>ci, axis=(0,1,2)) 
plt.plot(sig)
plt.axhline(thresh)
np.sum(sig>thresh)

_, bestlag = plot_stas(stas_[...,sig>thresh])

cids = np.where(sig>thresh)[0]

stas_fix = dsfix.get_stas(inds=ctr_inds, square=False)
stas_fix = stas_fix.detach().cpu().numpy()

#%%
from hires import sta_summary
contours = []
for i, cc in enumerate(cids):
    sum0 = sta_summary(stas_2[...,cc], int(bestlag[i]), label="Free Viewing", plot=True)
    contours.append(sum0['contour'])

# #%%

# stas_fixc = dsfixc.get_stas(square=False)

# #%%
# from hires import sta_summary
# stas_fixc2 = dsfixc.get_stas(square=True)
# contours = []
# for i, cc in enumerate(cids):
#     sum0 = sta_summary(stas_fixc2[...,cc], int(bestlag[i]), label="Free Viewing", plot=True)
#     contours.append(sum0['contour'])

#%% Compare STA on subsets of the data

from hires import sta_summary

''' Subsets of free-viewing data '''
data = ds[gab_inds]
datafix = dsfixc[:]

#%%

nsample = 1
np.random.seed(1234)
plot=False
NC = len(cids)
ccFVFV = np.zeros((nsample, NC))
ccFXFX = np.zeros((nsample, NC))
ccFVFX = np.zeros((nsample, NC))
ccFXFV = np.zeros((nsample, NC))
ccFXFXraw = np.zeros((nsample, NC))
thresh_ = np.ones((nsample,NC))


for b in range(nsample):
    print("sample %d/%d" %(b+1, nsample))
    n = data['robs'].shape[0]
    robs = data['robs']*data['dfs']
    inds = np.random.choice(n, size=n//2, replace=False) # bootstrap indices from FV data
    inds_pair = np.setdiff1d(np.arange(n), inds) # bootstrap indices from FV data for paired CC analysis
    nFV1 = len(inds)
    nFV2 = len(inds_pair)

    nyFV1 = robs[inds,:].sum(dim=0)
    sta_FV_sub1 = torch.einsum('ncxyt,nm->xytm', data['stim'][inds,...], robs[inds,:])
    sta_FV_sub1 /= nyFV1[None,None,None,:]
    sta_FV_sub1 = sta_FV_sub1.permute(2,0,1,3).detach().cpu().numpy()

    nyFV2 = robs[inds_pair,:].sum(dim=0)
    sta_FV_sub2 = torch.einsum('ncxyt,nm->xytm', data['stim'][inds_pair,...], robs[inds_pair,:])
    sta_FV_sub2 /= nyFV2[None,None,None,:]
    sta_FV_sub2 = sta_FV_sub2.permute(2,0,1,3).detach().cpu().numpy()

    ''' Subsets of fixation data '''
    n = datafix['robs'].shape[0]
    robs = datafix['robs']*datafix['dfs']

    inds = np.random.choice(n, size=n//2, replace=False) # bootstrap indices from FV datafix
    inds_pair = np.setdiff1d(np.arange(n), inds) # bootstrap indices from FV data for paired CC analysis

    nFX1 = len(inds)
    nFX2 = len(inds)

    nyFX1 = robs[inds,:].sum(dim=0)
    sta_FX_sub1 = torch.einsum('ncxyt,nm->xytm', datafix['stim'][inds,...], robs[inds,:])
    sta_FX_sub1 /= nyFX1[None,None,None,:]
    sta_FX_sub1 = sta_FX_sub1.permute(2,0,1,3).detach().cpu().numpy()

    nyFX2 = robs[inds_pair,:].sum(dim=0)
    sta_FX_sub2 = torch.einsum('ncxyt,nm->xytm', datafix['stim'][inds_pair,...], robs[inds_pair,:])
    sta_FX_sub2 /= nyFX2[None,None,None,:]
    sta_FX_sub2 = sta_FX_sub2.permute(2,0,1,3).detach().cpu().numpy()


    for i, cc in enumerate(cids):
        # try:
        blag = int(bestlag[i])
        sumFV1 = sta_summary(sta_FV_sub1[...,cc], blag, label="Free Viewing", plot=False, contour=contours[i])
        sumFV2 = sta_summary(sta_FV_sub2[...,cc], blag, label="Free Viewing", plot=False, contour=sumFV1['contour'])
        sumFX1 = sta_summary(sta_FX_sub1[...,cc], blag, label="Fixation (corrected)", plot=False, contour=sumFV1['contour'])
        sumFX2 = sta_summary(sta_FX_sub2[...,cc], blag, label="Fixation (corrected)", plot=False, contour=sumFV1['contour'])
        sumFXraw = sta_summary(stas_fix[...,cc], blag, label="Fixation (corrected)", plot=False, contour=sumFV1['contour'])

        rmask = sumFV1['rmask'].copy()
        ccFVFV[b,i] = np.corrcoef( sumFV1['Im'][rmask] , sumFV2['Im'][rmask] )[1,0]
        ccFXFX[b,i] = np.corrcoef( sumFX1['Im'][rmask] , sumFX2['Im'][rmask] )[1,0]
        ccFVFX[b,i] = np.corrcoef( sumFV1['Im'][rmask] , sumFX1['Im'][rmask] )[1,0]
        ccFXFV[b,i] = np.corrcoef( sumFV2['Im'][rmask] , sumFX2['Im'][rmask] )[1,0]
        thresh_[b,i] = sumFV1['thresh']
        
        ccFXFXraw[b,i] = np.corrcoef( sumFX1['Im'][rmask] , sumFXraw['Im'][rmask] )[1,0]

        if b == 0 and plot:
            plt.figure(figsize=(10, 5))
            plt.subplot(1,4,1)
            plt.imshow(sumFV1['Im'], interpolation='nearest', cmap=plt.cm.gray)
            plt.title("Free Viewing")
            plt.ylabel("Neuron %d" %cc)
            plt.subplot(1,4,2)
            plt.imshow(sumFV2['Im'], interpolation='nearest', cmap=plt.cm.gray)
            plt.title("Free Viewing (Pair) %.2f" %ccFVFV[b,i])
            plt.subplot(1,4,3)
            plt.imshow(sumFX1['Im'], interpolation='nearest', cmap=plt.cm.gray)
            plt.title("Fixation (Corrected) %.2f" %ccFVFX[b,i])
            plt.subplot(1,4,4)
            plt.imshow(sumFX2['Im'], interpolation='nearest', cmap=plt.cm.gray)
            plt.title("Fixation %.2f" %ccFXFX[b,i])
            plt.show()
        # except:
        #     pass

#%%






#%%
ccFXFXmu = np.mean(ccFXFX, axis=0)
ccFXFVmu = np.mean(np.concatenate([ccFVFX, ccFXFV], axis=0), axis=0)
ccFXFXrawmu = np.mean(ccFXFXraw, axis=0)
ccFVFVmu = np.mean(ccFVFV, axis=0)

inds = np.argsort(ccFXFXmu)
example_cells = [inds[-1], inds[31]] #int(.65*NC)


stas_fv = (sta_FV_sub1 + sta_FV_sub2)/2
stas_fx = (sta_FX_sub1 + sta_FX_sub2)/2

cc = example_cells[0]
blag = int(bestlag[cc])
sum0 = sta_summary(stas_fv[...,cids[cc]], blag, label="Free Viewing", plot=False)
sumfxc = sta_summary(stas_fx[...,cids[cc]], blag, label="Fixation (Corrected)", plot=False, contour=sum0['contour'])
sumfx = sta_summary(stas_fix[...,cids[cc]], blag, label="Fixation", plot=False, contour=sum0['contour'])
np.corrcoef( sum0['Im'][sum0['rmask']] , sumfxc['Im'][sum0['rmask']] )[1,0]
plt.subplot(2,3,1)
plt.imshow(sum0['Im'], interpolation='nearest', cmap=plt.cm.gray)
plt.plot(sum0['contour'][:,1], sum0['contour'][:,0], 'r')
plt.title("Free Viewing")
plt.ylabel("Example Cell 1")
plt.subplot(2,3,2)
plt.imshow(sumfxc['Im'], interpolation='nearest', cmap=plt.cm.gray)
plt.plot(sum0['contour'][:,1], sum0['contour'][:,0], 'r')
plt.title("Fixation (Corrected)")
plt.subplot(2,3,3)
plt.imshow(sumfx['Im'], interpolation='nearest', cmap=plt.cm.gray)
plt.plot(sum0['contour'][:,1], sum0['contour'][:,0], 'r')
plt.title("Fixation")

# cc = example_cells[1]
cc = example_cells[1]
blag = int(bestlag[cc])
sum0 = sta_summary(stas_fv[...,cids[cc]], blag, label="Free Viewing", plot=False)
sumfxc = sta_summary(stas_fx[...,cids[cc]], blag, label="Fixation (Corrected)", plot=False, contour=sum0['contour'])
sumfx = sta_summary(stas_fix[...,cids[cc]], blag, label="Fixation", plot=False, contour=sum0['contour'])

plt.subplot(2,3,4)
plt.imshow(sum0['Im'], interpolation='nearest', cmap=plt.cm.gray)
plt.plot(sum0['contour'][:,1], sum0['contour'][:,0], 'g')
plt.title("Free Viewing")
plt.ylabel("Example Cell 2")
plt.subplot(2,3,5)
plt.imshow(sumfxc['Im'], interpolation='nearest', cmap=plt.cm.gray)
plt.plot(sum0['contour'][:,1], sum0['contour'][:,0], 'g')
plt.title("Fixation (Corrected)")
plt.subplot(2,3,6)
plt.imshow(sumfx['Im'], interpolation='nearest', cmap=plt.cm.gray)
plt.plot(sum0['contour'][:,1], sum0['contour'][:,0], 'g')
plt.title("Fixation")

plt.savefig(os.path.join(fig_dir, 'FXFV_examples.pdf'))

plt.figure(figsize=(8,2.5))

plt.subplot(1,3,1)
# plt.plot(np.mean(ccFXFX, axis=0), np.mean(ccFXFV, axis=0), '.')
plt.plot(ccFXFXmu, ccFXFVmu, '.k')
plt.plot(ccFXFXmu[example_cells[0]], ccFXFVmu[example_cells[0]], 'or')
plt.plot(ccFXFXmu[example_cells[1]], ccFXFVmu[example_cells[1]], 'og')
from scipy.stats import mannwhitneyu
res = mannwhitneyu(ccFXFXmu, ccFXFVmu)
print("Compare FXFX and FXFV")
print(res)

plt.plot([0,1], [0,1], 'k')
plt.xlabel('Corr. (FX-FX)')
plt.ylabel('Corr. (FX-FV)')

plt.subplot(1,3,2)
plt.plot(ccFXFXmu, ccFXFXrawmu, '.k')
plt.plot(ccFXFXmu[example_cells[0]], ccFXFXrawmu[example_cells[0]], 'or')
plt.plot(ccFXFXmu[example_cells[1]], ccFXFXrawmu[example_cells[1]], 'og')
plt.plot([0,1], [0,1], 'k')
plt.xlabel('Corr. (FX-FX)')
plt.ylabel('Corr. (FX-FXraw)')
res = mannwhitneyu(ccFXFXmu, ccFXFXrawmu)
print("Compare FXFX and FXFXraw")
print(res)

plt.subplot(1,3,3)
plt.plot(ccFXFXmu, ccFVFVmu, '.k')
plt.plot(ccFXFXmu[example_cells[0]], ccFVFVmu[example_cells[0]], 'or')
plt.plot(ccFXFXmu[example_cells[1]], ccFVFVmu[example_cells[1]], 'og')
plt.plot([0,1], [0,1], 'k')
plt.xlabel('Corr. (FX-FX)')
plt.ylabel('Corr. (FV-FV)')
res = mannwhitneyu(ccFXFXmu, ccFVFVmu)
print("Compare FXFX and FVFV")
print(res)


import seaborn as sns
sns.despine(trim=True, offset=-10)

plt.savefig(os.path.join(fig_dir, 'FXFV_summary.pdf'))