

#%% Import Libraries
import sys
import os

import numpy as np

# Import torch
import torch


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
dataset_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

NUM_WORKERS = int(os.cpu_count() / 2)

%load_ext autoreload
%autoreload 2


#%% Load dataset

datadir = '/home/jake/Data/Datasets/Conway/pilot/'
fn = 'Jacomo230331pytorch.mat'

#%% Download processed dataset if needed
download = False
if download:
    import time

    def reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                        (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()

    import urllib

    url = 'https://www.dropbox.com/s/yaogelll0393trl/Jacomo230331pytorch.mat?dl=1'
    urllib.request.urlretrieve(url, datadir+fn, reporthook)

#%% Load data
import h5py
f = h5py.File(datadir + fn, 'r')
stim = f['stim'][:,:]
robs = f['robs'][:,:].T
eyepos = f['eyepos'][:,:].T

f.close()

#%% quick check
lag = 2
NT = stim.shape[0]

# restrict indices because these data are huge
fix_mask = np.hypot(eyepos[:,0], eyepos[:,1]) < 30
fix_good = np.where(fix_mask)[0]
fix_good = fix_good[fix_good < 935*240]
fix_good = fix_good[fix_good < NT-lag]

sta = stim[fix_good,0,:,:].reshape(-1, 60*60).T@robs[fix_good+lag,:]

NC = robs.shape[1]
sx = int(np.ceil(np.sqrt(NC)))
sy = int(np.ceil(NC/sx))
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for cc in range(NC):
    plt.subplot(sx,sy,cc+1)
    plt.imshow(sta[:,cc].reshape(60,60), cmap='gray')




#%% embed time
num_lags = 8
NT = stim.shape[0]
X = stim[np.arange(NT)[:,None]-np.arange(num_lags), ...].transpose(0,2,3,4,1)


#%% save torch dataset
data = {'stim': torch.from_numpy(X[fix_good,...].astype(np.float32)), 
        'robs': torch.from_numpy(robs[fix_good,:].astype(np.float32)),
        'eyepos': torch.from_numpy(eyepos[fix_good,:].astype(np.float32))}

torch.save(data, datadir + 'data.pt')

# %%
