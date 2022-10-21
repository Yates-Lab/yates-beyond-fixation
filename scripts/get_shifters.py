
#%% Import Libraries
import sys
import os


# setup paths
# 20200304

sys.path.insert(0, '/home/jake/Data/Repos/')
sys.path.insert(0, '/home/jake/Data/Repos/yates-beyond-fixation/scripts/')



import numpy as np

# plotting
import matplotlib.pyplot as plt

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
from datasets.pixel import Pixel
from models.utils import plot_stas

from datasets.pixel.utils import get_stim_list
sesslist = list(get_stim_list().keys())
sessname = sesslist[12]
datadir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'
NBname = 'shifter_{}'.format(sessname)
dirname = os.path.join('.', 'checkpoints', NBname)
print(dirname)
#%%
valid_eye_rad = 9.2
ds = Pixel(datadir,
    sess_list=[sessname],
    requested_stims=['Dots', 'Gabor'],
    num_lags=12,
    downsample_t=2,
    download=True,
    valid_eye_rad=valid_eye_rad,
    ctr=np.array([0,0]),
    fixations_only=True,
    load_shifters=False,
    spike_sorting='kilowf',
    covariate_requests={
        'fixation_onset': {'tent_ctrs': np.arange(-5, 40, 3)},
        'frame_tent': {'ntents': 20}}
    )
    
print("calculating datafilters")
ds.compute_datafilters(to_plot=True)
print("Done")
from copy import deepcopy
dims_orig = deepcopy(ds.dims)


#%% get STA on the Gabor stimulus
gab_inds = np.where(np.in1d(ds.valid_idx, ds.stim_indices[ds.sess_list[0]]['Gabor']['inds']))[0].tolist()

# stas0 = ds.get_stas(inds=gab_inds, square=False)
# stas = ds.get_stas(inds=gab_inds, square=True)

stas0 = ds.get_stas(square=False)
stas = ds.get_stas(square=True)

#%%
FracDF_include = .2
%matplotlib inline
cids = np.where(ds.covariates['dfs'].sum(dim=0) / ds.covariates['dfs'].shape[0] > FracDF_include)[0]
# cids = np.intersect1d(cids, np.where(ds.covariates['robs'].sum(dim=0)>1500)[0])

mu, bestlag = plot_stas(stas0[...,cids].detach().numpy())
mu, bestlag = plot_stas(stas[...,cids].detach().numpy())


#%% Find position on screen that has the most samples (resting gaze of the subject)
rad = 1 # window around each position to count samples
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

ctr_inds = np.where(np.hypot(ds.covariates['eyepos'][ds.valid_idx,0].cpu().numpy() - rgx0, ds.covariates['eyepos'][ds.valid_idx,1].cpu().numpy() - rgy0) < rad)[0].tolist()
print("computing STAs with %d indices" %len(ctr_inds))

#%% get STA in the center region
ctr_inds = np.where(np.hypot(ds.covariates['eyepos'][ds.valid_idx,0].cpu().numpy() - rgx0, ds.covariates['eyepos'][ds.valid_idx,1].cpu().numpy() - rgy0) < rad)[0].tolist()
# gab_inds = np.where(np.in1d(ds.valid_idx, ds.stim_indices[ds.sess_list[0]]['Gabor']['inds']))[0].tolist()
stas = ds.get_stas(inds=ctr_inds, square=True)

mu, bestlag = plot_stas(stas[...,cids].detach().numpy())

#%% Find spatial window to crop so the dataset will fit in GPU memory
# fitting on the GPU is WAY faster than transfering from CPU to GPU
win_size = 50

spower = stas[...,cids].std(dim=0)
spatial_power = torch.einsum('whn,n->wh', spower, ds.covariates['robs'][:,cids].sum(dim=0)/ds.covariates['robs'][:,cids].sum())
spatial_power[spatial_power < .5*spatial_power.max()] = 0 # remove noise floor
spatial_power /= spatial_power.sum()

xx,yy = torch.meshgrid(torch.arange(0, stas.shape[1]), torch.arange(0, stas.shape[2]))

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

stas2 = ds.get_stas(inds=ctr_inds, square=True)

# re-compute mus referenced in the cropped window size
mu, bestlag = plot_stas(stas2[...,cids].detach().numpy())


#%% initialize shifter using spatial power of the STAs
# from hires import get_shifter_from_spatial_power
# smod = get_shifter_from_spatial_power(ds, cids=cids, bestlag=bestlag)

#%% get training / test data
from hires import get_train_val_data
rad = 3.5 # radius centered
ctr_inds = np.where(np.hypot(ds.covariates['eyepos'][ds.valid_idx,0].cpu().numpy() - rgx0, ds.covariates['eyepos'][ds.valid_idx,1].cpu().numpy() - rgy0) < rad)[0].tolist()
train_data, val_data, train_inds, val_inds = get_train_val_data(ds, force_data_on_device=True,
    ctr_inds=ctr_inds)

#%% Build datasets / dataloaders
from datasets.generic import GenericDataset
batch_size = 1000

train_ds = GenericDataset(train_data, device=dataset_device)
val_ds = GenericDataset(val_data, device=torch.device('cpu'))

from torch.utils.data import DataLoader

if dataset_device == torch.device('cpu'):
    print("Dataloaders are on the cpu")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=os.cpu_count()//2, pin_memory=True)
else:
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)


#%% Instantiate shifter model
from models.shifters import ShifterModel
from models.utils import eval_model
from NDNT.training import Trainer, EarlyStopping
from NDNT.utils import NDNutils, load_model

input_dims = [1, win_size, win_size] + [ds.num_lags]
num_filters = [6, 6, 5]
filter_width = [7, 5, 3]

data = next(iter(train_dl))

lambdas = [1e-4, 1e-3, 1e-2]
for irun in range(5):
    d2x = np.random.randint(len(lambdas))
    d2t = np.random.randint(len(lambdas))
    center = np.random.randint(len(lambdas))
    
    cr0 = ShifterModel(input_dims, NC=ds.NC,
        num_subunits=num_filters,
        filter_width=filter_width,
        cids=cids,
        drifter=False,
        noise_sigma=0,
        reg_vals={'d2x': d2x, 'd2t': d2t, 'center': center},
        reg_hidden=None,
        reg_readout={'l2':1e-6},
        modifiers = {'stimlist': ['frame_tent'],
                    'gain': [None],
                    'offset':[data['frame_tent'].shape[1]],
                    'stage': ["readout"],
                    'outdims': [len(cids)]}
                    )
    
    # initialize parameters
    if mu is not None:
        cr0.readout.mu.data[:,:] = torch.tensor(mu, dtype=torch.float32)

    cr0.readout.sigma.data.fill_(0.5)
    cr0.readout.sigma.requires_grad = True

    cr0.bias.data[:] = torch.log(torch.exp(ds.covariates['robs'][:,cr0.cids].mean(dim=0)*cr0.output_NL.beta) - 1)/cr0.output_NL.beta
    cr0.bias.requires_grad = False


    cr0.prepare_regularization()
 
    cr0.to(device)

    parameters = cr0.parameters()

    optimizer = torch.optim.Adam(parameters,
            lr=0.001, weight_decay=1e-3)

    earlystopping = EarlyStopping(patience=10, verbose=False)

    trainer = Trainer(cr0, optimizer=optimizer,
        device = device,
        max_epochs=500,
        verbose=1,
        dirpath = dirname,
        early_stopping=earlystopping)

    trainer.fit(cr0, train_dl, val_dl)


#%% Load best model
cr0 = load_model(dirname, version=None)

#%%
from NDNT.utils import get_fit_versions
vers = get_fit_versions(dirname, '')

plt.plot(vers['version_num'], vers['val_loss'], 'o')

#%% find best version within bounds
bnds = [20, np.inf]
vernum = np.asarray(vers['version_num'])
vloss = np.asarray(vers['val_loss'])
ix = np.where(np.logical_and(vernum > bnds[0], vernum < bnds[1]))[0]
plt.plot(vernum[ix], vloss[ix], '.')
id = np.argmin(vloss[ix])
bestver = vernum[ix[id]]

#%%
plt.figure()
ll0 = eval_model(cr0, val_dl)
plt.plot(ll0, '-o')
plt.show()

#%%
ws = cr0.core[0].get_weights()
_ = plot_stas(np.transpose(ws, (2,0,1,3)))
plt.show()

#%% check shift on training set
cr2 = deepcopy(cr0)
shift = cr2.shifter(train_data['eyepos'])
from datasets.pixel.utils import shift_im
stim = train_data['stim'].cpu().reshape(-1, win_size, win_size, ds.num_lags).permute(0, 3, 1, 2)
stimshift = shift_im(stim, shift)

sta1 = torch.einsum('nlxy,nc->lxyc', stimshift, train_data['robs'].cpu()-train_data['robs'].cpu().mean(dim=0))
_ = plot_stas(sta1.numpy())
# plt.plot(shift.detach().cpu().numpy()

#%% null out shifter and compare
cr3 = deepcopy(cr2)
cr3.shifter[0].weight.data[:] = 0
cr3.shifter[2].weight.data[:] = 0
cr3.shifter[2].bias.data[:] = 0

ll1 = eval_model(cr3, val_dl)
plt.plot(ll0, ll1, 'o')
plt.plot(plt.xlim(), plt.xlim(), '--')
plt.xlabel("shifter (bits/spike)")
plt.ylabel("no shifter (bits/spike)")

#%% refit over larger range of eye positions
from hires import get_train_val_data
rad = 10 # radius centered
ctr_inds = np.where(np.hypot(ds.covariates['eyepos'][ds.valid_idx,0].cpu().numpy() - rgx0, ds.covariates['eyepos'][ds.valid_idx,1].cpu().numpy() - rgy0) < rad)[0].tolist()
train_data, val_data, train_inds, val_inds = get_train_val_data(ds, force_data_on_device=True,
    ctr_inds=ctr_inds)

cr2.core[0].reg.vals['d2x'] = 0.001
cr2.core[0].reg.vals['d2t'] = 0.001
cr2.core[0].reg.vals['center'] = 0.001

cr2.prepare_regularization()
from datasets.generic import GenericDataset
batch_size = 1000

train_ds = GenericDataset(train_data, device=dataset_device)
val_ds = GenericDataset(val_data, device=torch.device('cpu'))

from torch.utils.data import DataLoader

if dataset_device == torch.device('cpu'):
    print("Dataloaders are on the cpu")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=os.cpu_count()//2, pin_memory=True)
else:
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)


parameters = cr2.parameters()
# parameters = []
# for name, m in cr2.named_parameters():
#     if 'shifter' in name: # or 'sigma' in name or 'mu' in name:
#         print(name)
#         parameters.append(m)

optimizer = torch.optim.Adam(parameters,
        lr=0.001)

earlystopping = EarlyStopping(patience=10, verbose=False)

trainer = Trainer(cr2, optimizer=optimizer,
    device = device,
    max_epochs=500,
    verbose=2,
    dirpath = dirname,
    early_stopping=earlystopping)

trainer.fit(cr2, train_dl, val_dl)

plt.figure()
ll0 = eval_model(cr2, val_dl)
plt.plot(ll0, '-o')
plt.show()

#%%
ws = cr2.core[0].get_weights()
_ = plot_stas(np.transpose(ws, (2,0,1,3)))
plt.show()

#%%
from datasets.pixel.utils import plot_shifter
_ = plot_shifter(cr0.shifter, title='shifter')

#%%
#%% check shift on training set
# cr2 = deepcopy(cr0)
shift = cr2.shifter(train_data['eyepos'])
from datasets.pixel.utils import shift_im
stim = train_data['stim'].cpu().reshape(-1, win_size, win_size, ds.num_lags).permute(0, 3, 1, 2)
stimshift = shift_im(stim, shift)

sta1 = torch.einsum('nlxy,nc->lxyc', stimshift, train_data['robs'].cpu()-train_data['robs'].cpu().mean(dim=0))
_ = plot_stas(sta1.numpy())

#%% set paths
outdir = datadir + 'shifters'
from NDNT.utils.NDNutils import ensure_dir
ofname = os.path.join(outdir, "cr2_" + sessname + ".pt")

#%% save model
ensure_dir(outdir)
torch.save(cr2, ofname)
print("Done saving model")

#%% load model
cr1 = torch.load(ofname)
#%% get shift in pixels, then convert back to full size and retrain shifter
# from NDNT.utils.DanUtils import grid2pixel, pixel2grid
from hires import StandaloneShifter
from datasets.generic import GenericDataset
from NDNT.utils.DanUtils import pixel2grid
# cr2 = deepcopy(cr0)
shift = cr2.shifter(ds.covariates['eyepos'])
pxshift = (win_size*(shift+1)-1)/2 - win_size/2

#%%
x = (2*pxshift[:,0])/ds.raw_dims[2]
y = (2*pxshift[:,1])/ds.raw_dims[1]

newshift = torch.cat((x[:,None],y[:,None]), dim=1).detach().clone()


plt.figure()
plt.plot(shift[:,0].detach().cpu(), x.detach().cpu(), '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
#%%
ds.crop_idx = [0,dims_orig[1],0,dims_orig[2]]
ds.shift = newshift.detach().clone()

rad = 6.5
ctr_inds = np.where(np.hypot(ds.covariates['eyepos'][ds.valid_idx,0].cpu().numpy() - rgx0, ds.covariates['eyepos'][ds.valid_idx,1].cpu().numpy() - rgy0) < rad)[0].tolist()

stas_shift = ds.get_stas(inds=ctr_inds, square=False)

%matplotlib inline
_ = plot_stas(stas_shift.detach().numpy())

#%%
# shifter = ds.get_shifters()
# shift = shifter[sessname](ds.covariates['eyepos'])

# ds.shift = shift.detach().clone()

# rad = 6.5
# ctr_inds = np.where(np.hypot(ds.covariates['eyepos'][ds.valid_idx,0].cpu().numpy() - rgx0, ds.covariates['eyepos'][ds.valid_idx,1].cpu().numpy() - rgy0) < rad)[0].tolist()

# stas_shift = ds.get_stas(inds=ctr_inds, square=False)

# %matplotlib inline
# _ = plot_stas(stas_shift[...,cids].detach().numpy())

# plt.savefig(sessname + '_stas_shift3.png')

#%%

shiftds = GenericDataset({'stim': ds.covariates['eyepos'][ds.valid_idx,:], 'shift': newshift[ds.valid_idx,:]}, device=dataset_device)
# split shiftds into train and valid
ntest = len(shiftds)//5
ntrain = len(shiftds) - ntest
train_ds, test_ds = torch.utils.data.random_split(shiftds, (ntrain, ntest))

from torch.utils.data import DataLoader
shifttrain_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
shifttest_dl = DataLoader(test_ds, batch_size=32, shuffle=True)

from NDNT.training import Trainer, EarlyStopping

smod = StandaloneShifter()
optimizer = torch.optim.Adam(smod.parameters(), lr=0.001)

earlystopping = EarlyStopping(patience=5, verbose=False)

trainer = Trainer(smod, optimizer=optimizer,
    device = device,
    max_epochs=200,
    verbose=1,
    dirpath = os.path.join('./checkpoints/standalone_shifter', 'smod'),
    early_stopping=earlystopping)

# fit
trainer.fit(smod, shifttrain_dl, shifttest_dl)



#%% 

shift = smod(ds.covariates['eyepos'])

ds.crop_idx = [0,dims_orig[1],0,dims_orig[2]]
# shift = cr2.shifter(ds.covariates['eyepos'])
ds.shift = shift.detach().clone()

rad = 6.5
ctr_inds = np.where(np.hypot(ds.covariates['eyepos'][ds.valid_idx,0].cpu().numpy() - rgx0, ds.covariates['eyepos'][ds.valid_idx,1].cpu().numpy() - rgy0) < rad)[0].tolist()

# ctr_inds = ctr_inds[:len(ctr_inds)//2]

stas_shift = ds.get_stas(inds=ctr_inds, square=False)

# stas_shift = ds.get_stas(inds=valid_shifts)

%matplotlib inline
_ = plot_stas(stas_shift[...,cids].detach().numpy())

plt.savefig(sessname + '_stas_shift3.png')


torch.save(cr2, sessname + '_cr3.pt')

#%% save shifter

outfile = os.path.join(datadir, 'shifter_cr3_' + sessname + '_' + ds.spike_sorting + '.p')

# load all shifters and save them
shifters = nn.ModuleList()
for vernum in [0]:
    shifters.append(smod.cpu())

shifter = smod
shifter.cpu()

print("saving file")

outdict = {'cids': cids, 'shifter': shifter, 'shifters': shifters, 'vernum': [0], 'vallos': [0], 'numlags': ds.num_lags, 'tdownsample': ds.downsample_t, 'lengthscale': 1}
import pickle
pickle.dump( outdict, open(outfile, "wb" ) )


# %%

