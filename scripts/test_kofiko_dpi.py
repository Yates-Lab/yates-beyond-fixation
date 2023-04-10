
#%% Import Libraries
import sys
import os

# setup paths
sys.path.insert(0, '/Users/jake/Documents/')
sys.path.insert(0, '/Users/jake/Documents/yates-beyond-fixation/scripts/')
fig_dir = '/Users/jake/Documents/yates-beyond-fixation/figures/conway_dpi_pilot'

import numpy as np

# plotting
import matplotlib.pyplot as plt
plt.style.use('plot_style.txt')

# Import torch
import torch
from torch import nn
import NDNT
import NTdatasets

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
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

datadir = '/Users/jake/Dropbox/Datasets/Conway/pilot/'

#%% run analyses: fit LNP and Energy Model
fn = 'Jocamo_230322_full_CC_ET1D_nofix_v08'
import NTdatasets.conway.cloud_datasets as datasets
data = datasets.ColorClouds(
    filenames=[fn], eye_config=0, drift_interval=16, which_stim='stimLP', 
    datadir=datadir, luminance_only=True, binocular=False,
    include_MUs=False)

#%%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sync_clocks import convertSamplesToTimestamps, alignTimestampToTimestamp

# Import dDPI Data
ddpi_path = Path(datadir) / '2023_03_22_output.csv'
ddpi_df = pd.read_csv(ddpi_path)

p1_x = ddpi_df.p1_x.to_numpy()
p1_y = ddpi_df.p1_y.to_numpy()
p4_x = ddpi_df.p4_x.to_numpy()
p4_y = ddpi_df.p4_y.to_numpy()

p1 = np.stack([p1_x, p1_y], axis=1)
p4 = np.stack([p4_x, p4_y], axis=1)

dpi = p1 - p4

t_dpi = ddpi_df.ts.to_numpy()
sync_dpi = ddpi_df.din.to_numpy()

# Import Plexon Data
plex_et_path = Path(datadir) / 'Jocamo_230322_full_CC_Eyetraces.mat'
plex_et = loadmat(plex_et_path)

sync_plex = plex_et['PlexET_ad'].T[:,0]
t_plex = np.arange(sync_plex.shape[0]) / 1000


# Align dDPI to Plexon
dpi_ts = convertSamplesToTimestamps(t_dpi, sync_dpi)
plex_ts = convertSamplesToTimestamps(t_plex, sync_plex)

slope, intercept = alignTimestampToTimestamp(dpi_ts, plex_ts, debug=True)

t_dpi_aligned = t_dpi * slope + intercept
plex_trial_start_times = t_plex[plex_et['trial_start_inds'][0,:]]
# %%



#%%
C = plt.hist2d(dpi[:,0], dpi[:,1], bins=(np.arange(20, 50, .1), np.arange(-10, 30, .1)))

plt.figure()
plt.imshow(np.log(C[0]+0.001), extent=[C[2][0], C[2][-1], C[1][0], C[1][-1]])

# %%

xx, yy = np.meshgrid(C[2][:-1], C[1][:-1])

x0 = np.sum(C[0]*xx)/np.sum(C[0])
y0 = np.sum(C[0]*yy)/np.sum(C[0])

print(x0, y0)

iTrial = 0
# %%
%matplotlib ipympl
iTrial += 1
trialDur = 3

ix = np.logical_and(t_dpi_aligned > plex_trial_start_times[iTrial], t_dpi_aligned < (plex_trial_start_times[iTrial]+trialDur))
t0 = t_dpi_aligned[ix][0]
plt.figure(figsize=(3,2))
plt.plot(t_dpi_aligned[ix]- t0, dpi[ix,0]-y0, linewidth=.5)
plt.plot(t_dpi_aligned[ix]-t0, dpi[ix,1]-x0, linewidth=.5)
plt.axhline(1, color='k', linewidth=.5)
plt.axhline(-1, color='k', linewidth=.5)
plt.ylim((-5, 5))
plt.show()
# %%
%matplotlib inline
plt.plot(ddpi)