
#%% Import Libraries
import sys
import os

# setup paths
sys.path.insert(0, '/home/jake/Data/Repos/')
sys.path.insert(0, '/home/jake/Data/Repos/yates-beyond-fixation/scripts/')
fig_dir = '/home/jake/Data/Repos/yates-beyond-fixation/figures/supp_simple_complex'

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

# Where saved models and checkpoints go -- this is to be automated
# print( 'Save_dir =', dirname)

NUM_WORKERS = int(os.cpu_count() / 2)

%load_ext autoreload
%autoreload 2



#%% Load dataset
# from datasets.pixel.utils import get_stim_list
# sesslist = list(get_stim_list().keys())
# for i,sess in enumerate(sesslist):
#     print('%d) %s' %(i,sess))

datadir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'


#%%

import h5py

fname = os.path.join(datadir, 'unitstats.mat')
# from datasets.utils import download_file
# download_file('https://www.dropbox.com/s/ybmlp4z4iud2tgi/unitstats.mat?dl=1', fname)

f = h5py.File(fname, 'r')
usesslist = list(f.keys())



#%% run analyses: fit LNP and Energy Model
import hires
sesslist = ['20191119', '20191121', '20191205', '20191206',  '20191231', '20200304', '20220601', '20220610']
for sessname in sesslist:
    
    print("Running analysis on %s" %sessname)
    ln_vs_en = hires.compare_ln_en(datadir, sessname)



# %% Load analyses
outdir = os.path.join(datadir, 'LNENmodels')

llLNP = []
llENP = []
SUindex = []
ws0 = []
wsEn1 = []
wsEn2 = []
wsEn3 = []
wsEn4 = []

fname = os.path.join(datadir, 'unitstats.mat')
fu = h5py.File(fname, 'r')
usesslist = list(fu.keys())

flist = os.listdir(outdir)
sesslist = [sess[9:17] for sess in flist]

import pickle

for i in range(len(sesslist)):
    sess = sesslist[i]
    fname = os.path.join(outdir, flist[i])

    usess = [s for s in usesslist if sess in s]
    if len(usess) ==0:
        continue
    else:
        usess = usess[0]

    with open(fname, 'rb') as f:
        ln_vs_en = pickle.load(f)

        
        cids = np.asarray(fu[usess]['cids'][:]).T
        isiV = fu[usess]['isiV'][:]
        uQ = fu[usess]['uQ'][:].T

        ii = np.where(np.logical_and(uQ > 25 , isiV < .2))[0]
        SUcids = cids[ii]

        cids = ln_vs_en['glm0'].cids # matlab is 1 based
        llLNP.append(ln_vs_en['llval0test'])
        llENP.append(ln_vs_en['llvalEntest'])
        SUindex.append(np.in1d(np.asarray(cids), np.asarray(SUcids)))
        ws0.append(ln_vs_en['ws0'])
        wsEn1.append(ln_vs_en['wsEn'][0])
        wsEn2.append(ln_vs_en['wsEn'][1])
        wsEn3.append(ln_vs_en['wsEn'][2])
        wsEn4.append(ln_vs_en['wsEn'][3])
    


# %%
import seaborn as sns



llsLNP = np.concatenate(llLNP)
llsENP = np.concatenate(llENP)
ii0 = np.logical_or(llsLNP>0, llsENP>0)
iix = np.logical_and( np.concatenate(SUindex), ii0)

iisimple = llsLNP > llsENP
is_simple_tot = llsLNP[ii0] > llsENP[ii0]
is_simple_SU = llsLNP[iix] > llsENP[iix]

LNw = np.concatenate(ws0, axis=-1)
ENw = []
ENw.append(np.concatenate(wsEn1, axis=-1))
ENw.append(np.concatenate(wsEn2, axis=-1))
ENw.append(np.concatenate(wsEn3, axis=-1))
ENw.append(np.concatenate(wsEn4, axis=-1))

plt.figure(figsize=(4,4))
plt.plot(llsLNP[ii0], llsENP[ii0], '.b')
plt.plot(llsLNP[np.logical_and(ii0,iisimple)], llsENP[np.logical_and(ii0,iisimple)], '.r')
# plt.plot(llsLNP[iix], llsENP[iix], '.')
xd = [-.1, .8]
plt.xlim(xd)
plt.ylim(xd)
plt.plot(plt.xlim(), plt.xlim(), 'k')
sns.despine(offset=10, trim=True)
plt.xlabel("LNP")
plt.ylabel("Energy Model")
plt.savefig(os.path.join(fig_dir, 'model_compare.pdf'))

#%%
# h1 = plt.hist(llsLNP[ii0]-llsENP[ii0], bins=np.arange(-2,2,.1))
# h2 = plt.hist(llsLNP[iix]-llsENP[iix], bins=h1[1])


d = llsLNP[ii0]-llsENP[ii0]
h1 = plt.hist(d, bins=np.arange(-1,1,.1), color='b')
h2 = plt.hist(d[is_simple_tot], bins=h1[1], color='r')
# h2 = plt.hist(llsLNP[iix]-llsENP[iix], bins=h1[1])

print("Fraction simple cells total: %d/%d (%2.2f)" %(np.sum(is_simple_tot), len(is_simple_tot) ,np.mean(is_simple_tot)))
print("Fraction simple cells SUs: %d/%d (%2.2f)" %(np.sum(is_simple_SU), len(is_simple_SU) ,np.mean(is_simple_SU)))
plt.savefig(os.path.join(fig_dir, 'model_compare_hist.pdf'))
# %% examples
from models.utils import plot_stas
simple_examples = np.where(np.logical_and(llsLNP > .1 , llsLNP > llsENP))[0]
complex_examples = np.where(np.logical_and(llsLNP > .2 , llsLNP < llsENP))[0]

cell_examples = np.concatenate((simple_examples, complex_examples))

mu, bestlag = plot_stas(LNw[..., cell_examples], plot=False)

# i = 6 # cell
# if i >= len(cell_examples):
#     i = 0

for i in range(len(cell_examples)):

    RF = LNw[bestlag[i],:,:,cell_examples[i]]
    vmax = np.max(np.abs(RF))
    imax,jmax = np.where(RF == np.max(RF))
    imin,jmin = np.where(RF == np.min(RF))
    lags = np.arange(1, LNw.shape[0]+1)/120*1000

    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    plt.imshow(RF, interpolation='none', cmap=plt.cm.gray, vmin=-vmax, vmax=vmax)
    plt.plot(jmax, imax, '.b')
    plt.plot(jmin, imin, '.r')
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.plot(lags, LNw[:,imax,jmax,cell_examples[i]], 'b')
    plt.plot(lags, LNw[:,imin,jmin,cell_examples[i]], 'r')
    plt.axhline(0, color='k')
    plt.xlabel("Time (ms)")
    sns.despine(offset=10, trim=True)
    plt.title("LN: %2.2f, EN: %2.2f" %(llsLNP[cell_examples[i]], llsENP[cell_examples[i]]))
    plt.savefig(os.path.join(fig_dir, 'example_%d_lin.pdf' %cell_examples[i]))

    plt.figure(figsize=(8,3*4))

    for q in range(4):
        mu, bestlag = plot_stas(ENw[q][..., cell_examples], plot=False)
        RF = ENw[q][bestlag[i],:,:,cell_examples[i]]
        vmax = np.max(np.abs(RF))
        imax,jmax = np.where(RF == np.max(RF))
        imin,jmin = np.where(RF == np.min(RF))
        lags = np.arange(1, ENw[q].shape[0]+1)/120*1000


        plt.subplot(4,2,q*2+1)
        plt.imshow(RF, interpolation='none', cmap=plt.cm.gray, vmin=-vmax, vmax=vmax)
        plt.plot(jmax, imax, '.b')
        plt.plot(jmin, imin, '.r')
        plt.axis("off")

        plt.subplot(4,2,q*2+2)
        plt.plot(lags, ENw[q][:,imax,jmax,cell_examples[i]], 'b')
        plt.plot(lags, ENw[q][:,imin,jmin,cell_examples[i]], 'r')
        plt.axhline(0, color='k')
        sns.despine(offset=10, trim=True)

    plt.xlabel("Time (ms)")
    plt.savefig(os.path.join(fig_dir, 'example_%d_quad.pdf' %cell_examples[i]))




# %%
