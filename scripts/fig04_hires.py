
#%% Import Libraries
import sys
import os



# setup paths
sys.path.insert(0, '/home/jake/Data/Repos/')
sys.path.insert(0, '/home/jake/Data/Repos/yates-beyond-fixation/scripts/')
fig_dir = '/home/jake/Data/Repos/yates-beyond-fixation/figures/fig04'

from NDNT.utils.NDNutils import ensure_dir
ensure_dir(fig_dir)

import numpy as np

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('plot_style.txt')

# Import torch
import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

NUM_WORKERS = int(os.cpu_count() / 2)

%load_ext autoreload
%autoreload 2



#%% Load datasets, do fitting
''' 
This requires A LOT of system memory to run ~200GB
'''

datadir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'

sesslist = ['20191119', '20191121', '20191205', '20191206',  '20191231', '20200304', '20220601', '20220610']

import hires
for i in range(len(sesslist)):
    rfsum = hires.fig04_rf_analysis(sesslist[i], datadir=datadir, overwrite=False)
    Nlin = sum(rfsum['rflin']['sig'] > rfsum['rflin']['thresh']*2)
    Nsquare = sum(rfsum['rfsquare']['sig'] > rfsum['rfsquare']['thresh']*2)
    print('%s\t%2.2f\t\t%d\t%d\t%d' %(rfsum['sessname'], rfsum['num_samples']/rfsum['frate'], Nlin, Nsquare, rfsum['NC']))


#%% plot examples
import seaborn as sns

examplesess = ['20200304', '20191121', '20191205', '20220610']
examplecells = [28, 11, 7, 108]

for i in range(len(examplesess)):

    rfsum = hires.fig04_rf_analysis(examplesess[i], datadir=datadir)    

    cc = examplecells[i]-1 # offset because python is 0-based index
    stans = rfsum['rflinns'][...,cc].numpy()
    sta = rfsum['rflin']['stas'][...,cc]
    stans = (stans - np.mean(stans)) / np.std(stans)
    sta = (sta - np.mean(sta)) / np.std(sta)

    vmax = np.max(np.abs(sta))

    extent = rfsum['rect'][[1,3,2,0]]/rfsum['ppd']*60
    extent[2:] = -extent[2:]
    numlags = sta.shape[0]
    plt.figure(figsize=(10,3))
    for ilag in range(numlags):
        plt.subplot(1,numlags, ilag+1)
        plt.imshow(np.flipud(np.rot90(sta[ilag,:,:], 1)), vmin=-vmax, vmax=vmax, interpolation='none', cmap='coolwarm', extent=extent, origin='lower')
        plt.axis("off")
    plt.savefig(os.path.join(fig_dir, 'example%s_%d_time.pdf' %(examplesess[i], examplecells[i])))

    plt.figure()
    blag = rfsum['rflin']['bestlag'][cc]
    plt.imshow(np.flipud(np.rot90(sta[blag,:,:], 1)), vmin=-vmax, vmax=vmax, interpolation='none', cmap='coolwarm', extent=extent, origin='lower')
    plt.grid(True)
    plt.colorbar()
    sns.despine(offset=0, trim=False)
    plt.savefig(os.path.join(fig_dir, 'example%s_%d_shift.pdf' %(examplesess[i], examplecells[i])))

    plt.figure()
    plt.imshow(np.flipud(np.rot90(stans[blag,:,:], 1)), vmin=-vmax, vmax=vmax, interpolation='none', cmap='coolwarm', extent=extent, origin='lower')
    plt.grid(True)
    plt.colorbar()
    sns.despine(offset=0, trim=False)
    plt.savefig(os.path.join(fig_dir, 'example%s_%d_noshift.pdf' %(examplesess[i], examplecells[i])))

#%%
contourx = []
contoury = []
ctx = []
cty = []
area = []
threshs = []
sessnum = []
sum0 = []
nsig = []
ntot = []

for isess in range(len(sesslist)):
    rfsum = hires.fig04_rf_analysis(sesslist[isess], datadir=datadir)

    field = 'rfsquare'
    cids = np.where(rfsum[field]['sig'] > 0.002)[0]
    
    ppd = rfsum['ppd']
    rect = rfsum['rect']
    extent = [rect[0]/ppd, rect[2]/ppd, rect[3]/ppd, rect[1]/ppd]

    NC = len(cids)
    nsig.append(NC)
    ntot.append(len(rfsum[field]['sig']))
    sx = int(np.ceil(np.sqrt(NC)))
    sy = int(np.round(np.sqrt(NC)))
    plt.figure(figsize=(sx,sy))
    for i, cc in enumerate(cids):
        sum0.append(hires.sta_summary(rfsum[field]['stas'][...,cc], int(rfsum[field]['bestlag'][cc]), label="Free Viewing", plot=False))
        
        plt.subplot(sx, sy, i + 1)
        plt.imshow(sum0[-1]['Im'].T, interpolation='none', extent=extent, cmap=plt.cm.gray)
        cx = sum0[-1]['center'][0]/ppd+rfsum['extent'][0]
        cy = sum0[-1]['center'][1]/ppd+rfsum['extent'][3]
        ar = sum0[-1]['area']/ppd/ppd

        contourx.append(sum0[-1]['contour'][:,0]/ppd+extent[0])
        contoury.append(sum0[-1]['contour'][:,1]/ppd+extent[3])
        ctx.append(sum0[-1]['center'][0]/ppd+extent[0])
        cty.append(sum0[-1]['center'][1]/ppd+extent[3])
        area.append(sum0[-1]['area']/ppd/ppd)
        threshs.append(sum0[-1]['thresh'])
        sessnum.append(isess)

        plt.plot( sum0[-1]['contour'][:,0]/ppd+extent[0], sum0[-1]['contour'][:,1]/ppd+extent[3], 'r', linewidth=.25)
        plt.axis("tight")

ampshift = []
ampnoshift = []
threshlin = []
nsiglin = []
for isess in range(len(sesslist)):
    rfsum = hires.fig04_rf_analysis(sesslist[isess], datadir=datadir)

    field = 'rflin'
    cids = np.where(rfsum[field]['sig'] > 0.002)[0]
    nsiglin.append(len(cids))
    
    for cc in cids:
        # get amplitude with and without shift
        sta = rfsum['rflin']['stas'][...,cc]
        sta = (sta - np.mean(sta)) / np.std(sta)
        stans = rfsum['rflinns'][...,cc].numpy()
        stans = (stans - np.mean(stans)) / np.std(stans)
        blag = int(rfsum['rflin']['bestlag'][cc])
        sta = sta[blag,...]
        stans = stans[blag,...]
        summ = hires.sta_summary(rfsum['rflin']['stas'][...,cc], int(rfsum['rflin']['bestlag'][cc]), label="Free Viewing", plot=False)
        threshlin.append(summ['thresh'])
        ampshift.append(np.max(sta) - np.min(sta))
        ampnoshift.append(np.max(stans)-np.min(stans))

#%%
print("rf square: %d /%d (%2.2f)" %(sum(nsig), sum(ntot), 100*sum(nsig) / sum(ntot)))
print("rf lin: %d /%d (%2.2f)" %(sum(nsiglin), sum(ntot), 100*sum(nsiglin) / sum(ntot)))

iix = np.where(np.asarray(threshlin) < 1)[0]
ampnoshift = np.asarray(ampnoshift)
ampshift = np.asarray(ampshift)

from scipy.stats import ttest_1samp
plt.figure()
plt.plot(ampnoshift[iix], ampshift[iix], 'o')
plt.xlabel("no shift")
plt.ylabel("shift")
plt.plot(plt.xlim(), plt.xlim(), 'k')

def geomeanci(x):

    n = len(x)
    logx = np.log(x)
    log_gm = np.mean(logx)
    log_se = np.std(logx)/np.sqrt(n)
    log_ci_95 = np.asarray([log_gm-2*log_se, log_gm+2*log_se])
    
    # exponentiate to get back to the geomean / se
    gm = np.exp(log_gm)
    ci_95 = np.exp(log_ci_95)
    return gm, ci_95



xrat = np.asarray(ampshift)/np.asarray(ampnoshift)
xrat = xrat[iix]

gm, ci = geomeanci(xrat)
print("%2.2f [%2.2f, %2.2f] n=%d" %(gm, ci[0], ci[1], len(xrat)))

res = ttest_1samp(xrat, 0)
print(res)

# np.log(np.asarray(ampshift)) - nnp.asarray(ampnoshift)


#%%
iix = np.where(np.asarray(threshs) < .51)[0]
ecc = np.hypot(np.asarray(ctx), np.asarray(cty))
ar = np.asarray(area)
for isess in np.unique(sessnum):
    ii = np.intersect1d(np.where(isess==np.asarray(sessnum))[0], iix)
    plt.plot(ecc[ii], ar[ii], '.', label=sesslist[isess])

# ar = np.asarray([np.sum(s['rmask']) for s in sum0])[iix]/ppd/ppd
# plt.plot(ecc, ar, '.')
plt.legend()
plt.ylim([0,.4])



#%%
iix = np.where(np.asarray(threshs) < .71)[0]
ecc = np.hypot(np.asarray(ctx), np.asarray(cty))[iix]
ar = np.asarray(area)[iix]
plt.plot(ecc, ar, '.')
plt.ylim([0,.4])

from scipy.io import savemat
savemat(os.path.join(fig_dir, 'eccentricitydata.mat'), {'ecc': ecc, 'area': ar})


#%%
plt.figure()
for i in range(len(area)):
    plt.plot(contourx[i], -contoury[i], alpha=.1)


#%%

import h5py

fname = os.path.join(datadir, 'unitstats.mat')
# from datasets.utils import download_file
# download_file('https://www.dropbox.com/s/ybmlp4z4iud2tgi/unitstats.mat?dl=1', fname)

f = h5py.File(fname, 'r')
usesslist = list(f.keys())

#%%
from datasets.pixel import Pixel
from models.utils import plot_stas

sessname = '20220610'
datadir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'

valid_eye_rad = 9.2
spike_sorting = 'kilowf'
try:
    ds = Pixel(datadir,
        sess_list=[sessname],
        requested_stims=['Dots', 'Gabor'],
        num_lags=12,
        downsample_t=2,
        download=True,
        valid_eye_rad=valid_eye_rad,
        ctr=np.array([0,0]),
        fixations_only=True,
        load_shifters=True,
        spike_sorting=spike_sorting,
        )
except:
    spike_sorting = 'kilo'
    ds = Pixel(datadir,
        sess_list=[sessname],
        requested_stims=['Dots', 'Gabor'],
        num_lags=12,
        downsample_t=2,
        download=True,
        valid_eye_rad=valid_eye_rad,
        ctr=np.array([0,0]),
        fixations_only=True,
        load_shifters=True,
        spike_sorting=spike_sorting,
        )
    
print("Done")
from copy import deepcopy
dims_orig = deepcopy(ds.dims)

#%% Find position on screen that has the most samples (resting gaze of the subject)
rad = 2 # window around each position to count samples
xax = np.arange(-(valid_eye_rad-1), (valid_eye_rad-1), .5) # grid up space

xx,yy = np.meshgrid(xax, xax)
dims = xx.shape
xx = xx.flatten()
yy = yy.flatten()
n = len(xx)

ns = np.zeros(n)
for i in range(n):
    # try:
    x0 = xx[i]
    y0 = yy[i]

    dist = np.hypot(ds.covariates['eyepos'][ds.valid_idx,0].cpu().numpy() - x0, ds.covariates['eyepos'][ds.valid_idx,1].cpu().numpy() - y0)
    
    inds = np.where(dist <= rad)[0]
    ns[i] = len(inds)

id = np.argmax(ns)
# resting gaze cetner
rgx0 = xx[id]
rgy0 = yy[id]

print("Found %d samples at %.3f, %.3f" %(ns[id], rgx0, rgy0))

plt.figure()
plt.imshow(ns.reshape(dims), cmap='coolwarm', extent=[-valid_eye_rad, valid_eye_rad, valid_eye_rad, -valid_eye_rad])
ax = plt.gca()
ax.invert_yaxis()
plt.xlabel('horizontal position (deg)')
plt.ylabel('vertical position (deg)')
plt.plot(rgx0, rgy0, 'o', color='k')

rad = 6
ctr_inds = np.where(np.hypot(ds.covariates['eyepos'][ds.valid_idx,0].cpu().numpy() - rgx0, ds.covariates['eyepos'][ds.valid_idx,1].cpu().numpy() - rgy0) < rad)[0].tolist()
print("computing STAs with %d indices" %len(ctr_inds))


#%% Get STA


data = ds[ctr_inds]

#%%
def get_stas_sig(data, thresh=0.001):

    robs = data['robs']*data['dfs']

    ny = robs.sum(dim=0)

    stas = torch.einsum('ncxyt,nm->xytm', data['stim'], robs)
    stas /= ny[None,None,None,:]
    stas = stas.permute(2,0,1,3).detach().cpu().numpy()

    num_lags = data['stim'].shape[-1]
    
    # get null sta by offsetting spikes acausally by num_lags
    stas_null = torch.einsum('ncxyt,nm->xytm', data['stim'][num_lags*2:,...], robs[:-num_lags*2,:])
    stas_null /= robs[:-num_lags*2,:].sum(dim=0)[None,None,None,:]
    stas_null = stas_null.permute(2,0,1,3).detach().cpu().numpy()


    mu = np.mean(stas_null, axis=0)
    thigh = np.quantile( (stas_null-mu[None,:,:,:]), 1-(thresh/2/num_lags), axis=(0,1,2))
    tlow = np.quantile( (stas_null-mu[None,:,:,:]), thresh/2/num_lags, axis=(0,1,2))

    sig = np.mean((stas - mu[None,:,:,:])>thigh[None,None,None,:], axis=(0,1,2))
    sig = sig + np.mean((stas - mu[None,:,:,:])<tlow[None,None,None,:], axis=(0,1,2))
    plt.plot(sig, '-o')
    plt.axhline(thresh*2)

    s = stas - mu[None,:,:,:]
    
    _, bestlag = plot_stas(s, plot=False)

    num_sig = np.sum(sig > thresh*2)
    print('%d/%d significant RFs' %(num_sig, len(sig)))

    return {'stas': s, 'sig': sig, 'thresh': thresh, 'nspikes': ny, 'bestlag':bestlag}

def get_stas_sig_square(data, thresh=0.001):

    robs = data['robs']*data['dfs']

    ny = robs.sum(dim=0)

    stas = torch.einsum('ncxyt,nm->xytm', data['stim']**2, robs)
    stas /= ny[None,None,None,:]
    stas = stas.permute(2,0,1,3).detach().cpu().numpy()

    num_lags = data['stim'].shape[-1]
    
    # get null sta by offsetting spikes acausally by num_lags
    stas_null = torch.einsum('ncxyt,nm->xytm', data['stim'][num_lags*2:,...]**2, robs[:-num_lags*2,:])
    stas_null /= robs[:-num_lags*2,:].sum(dim=0)[None,None,None,:]
    stas_null = stas_null.permute(2,0,1,3).detach().cpu().numpy()


    mu = np.mean(stas_null, axis=0)
    thigh = np.quantile( (stas_null-mu[None,:,:,:]), 1-(thresh/2/num_lags), axis=(0,1,2))
    tlow = np.quantile( (stas_null-mu[None,:,:,:]), thresh/2/num_lags, axis=(0,1,2))

    sig = np.mean((stas - mu[None,:,:,:])>thigh[None,None,None,:], axis=(0,1,2))
    sig = sig + np.mean((stas - mu[None,:,:,:])<tlow[None,None,None,:], axis=(0,1,2))
    plt.plot(sig, '-o')
    plt.axhline(thresh*2)

    s = stas - mu[None,:,:,:]
    
    _, bestlag = plot_stas(s, plot=False)

    num_sig = np.sum(sig > thresh*2)
    print('%d/%d significant RFs' %(num_sig, len(sig)))

    return {'stas': s, 'sig': sig, 'thresh': thresh, 'nspikes': ny, 'bestlag':bestlag}

rf = get_stas_sig(data)
rf2  = get_stas_sig_square(data)
#%%

cc += 1
plt.subplot(1,2,1)
plt.imshow(rf2['stas'][rf2['bestlag'][cc],:,:,cc])
plt.subplot(1,2,2)
plt.imshow(rf['stas'][rf2['bestlag'][cc],:,:,cc])


#%% fit gaussian to spatial RF
from scipy.optimize import curve_fit
import scipy.optimize as opt
isess = 0
rfsum = hires.fig04_rf_analysis(sesslist[isess], datadir=datadir)

field = 'rfsquare'
cids = np.where(rfsum[field]['sig'] > 0.002)[0]

sum0 = []
for i, cc in enumerate(cids):
    sum0.append(hires.sta_summary(rfsum[field]['stas'][...,cc], int(rfsum[field]['bestlag'][cc]), label="Free Viewing", plot=False))


#%%
from datasets.utils import r_squared
centerx = []
centery = []
sdopt = []
ampopt = []
r2s = []

def gauss2D(xytuple, x0, y0, sd, base, amplitude):
    (x,y) = xytuple
    y = base + amplitude * np.exp( -.5 * ((x - x0)**2 + (y - y0)**2 ) / sd**2 )
    
    return y.ravel()

for cc in range(len(cids)):

    Im = sum0[cc]['Im'].T
    Im = Im.astype('float64')
    sz = Im.shape
    xax = np.linspace(rfsum['extent'][0], rfsum['extent'][1], sz[1])
    yax = np.linspace(rfsum['extent'][2], rfsum['extent'][3], sz[0])
    xx,yy = np.meshgrid(xax, yax)

    ppd = rfsum['ppd']
    cx = sum0[cc]['center'][0]/ppd+rfsum['extent'][0]
    cy = sum0[cc]['center'][1]/ppd+rfsum['extent'][3]
    ar = sum0[cc]['area']/ppd/ppd
    amp = np.max(Im)
    extent = rfsum['extent']


    x = np.concatenate((xx.flatten()[:,None], yy.flatten()[:,None]), axis=1).astype('float64')
    sd = np.sqrt(ar).astype('float64')

    y = gauss2D((xx,yy), cx, cy, sd, 0, 1)
    plt.figure()
    ax = plt.axes()
    # Im[Im < .2*amp] = 0 
    ax.imshow(Im, interpolation='none', extent=extent, cmap=plt.cm.gray, origin='lower')
    ax.plot( sum0[cc]['contour'][:,0]/ppd+extent[0], sum0[cc]['contour'][:,1]/ppd+extent[3], 'r', linewidth=.25)
    # plt.axis("tight")
    ax.plot(cx, cy, '.r')

    # ax.imshow(y.reshape(sz), interpolation='none', extent=extent, cmap=plt.cm.gray)
    # ax.plot( sum0[cc]['contour'][:,0]/ppd+extent[0], sum0[cc]['contour'][:,1]/ppd+extent[3], 'r', linewidth=.25)

    # ax.plot(cx, cy, '.r')


    # if -np.min(Im) > amp:
    #     amp = np.min(Im)

    popt, pcov = curve_fit(gauss2D, (xx,yy), Im.flatten(), p0=(cx, cy, sd, 0, amp))
    # popt = (cx, cy, sd, 0, 1)
    data_fitted = gauss2D((xx, yy), *popt)

    ax.contour(xx, yy, data_fitted.reshape(sz), 4, colors='r', linewidth=.5)
    # ax.plot(popt[0], popt[1], 'go')
    plt.show()

    centerx.append(popt[0])
    centery.append(popt[1])
    sdopt.append(popt[2])
    ampopt.append(popt[-1])
    r2s.append(r_squared(Im.flatten()[:,None], data_fitted.flatten()[:,None]))

#%%

iix = np.where(np.asarray(r2s) > .5)[0]
ecc = np.hypot(np.asarray(centerx)[iix], np.asarray(centery)[iix])
ar = np.asarray(sdopt)[iix]

plt.plot(ecc, ar, '.')