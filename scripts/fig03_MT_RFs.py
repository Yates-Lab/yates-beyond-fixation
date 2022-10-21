#%% 
import sys, os

sys.path.insert(0, '/home/jake/Data/Repos/')
sys.path.insert(0, '/home/jake/Data/Repos/yates-beyond-fixation/scripts/')

datadir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'
dirname = os.path.join('.', 'checkpoints')

import numpy as np
import matplotlib.pyplot as plt  # plotting

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_WORKERS = int(os.cpu_count() / 2)

%load_ext autoreload
%autoreload 2

#%%

from datasets.mtdots import MTDotsDataset, get_stim_file
dirpath = '/mnt/Data/Datasets/MitchellV1FreeViewing/MT_RF/'
stim_list = get_stim_file(None)
sess_list = list(stim_list.keys())
# %% main analysis

def fit_session(sessname, dirpath, overwrite=False):
    import pickle
    from NDNT.utils.plotting import plot_filters_ST3D

    fname = '../output/fig03_MT_RFs/glm_%s.pkl' % sessname
    if os.path.isfile(fname) and not overwrite:
        with open(fname, 'rb') as handle:
            sessfit = pickle.load(handle)
            return sessfit
         
    mt = MTDotsDataset(sessname, dirpath)
    print("%s: %d spikes" %(sessname,mt.robs.sum()))

    # get STA and see if there's any structure
    print("plotting STAS...")
    train_data = mt[:]
    ws = (train_data['stim'].T@train_data['robs'])/train_data['robs'].sum(dim=0)
    w = ws.reshape(mt.dims + [-1]).detach().cpu().numpy()
    plot_filters_ST3D(w)

    
    NT = len(mt)

    from torch.utils.data import random_split
    train_ds, test_ds = random_split(mt, [NT-NT//5, NT//5])

    val_data = test_ds[:]
    train_data = train_ds[:]

    from models.glm import GLM
    from models.utils import eval_model
    from NDNT.training import LBFGSTrainer


    input_dims = list(mt.dims)

    cids = np.where(mt.robs.sum(dim=0)>1000)[0]
    d2x = 100
    d2t = 100
    loc = 100

    glm0 = GLM(input_dims=input_dims,
            cids=cids,
            reg_vals={'d2x':d2x,
            'd2t': d2t,
            'glocalx': loc})

    glm0.bias.data = torch.log(torch.exp(train_data['robs'][:,glm0.cids].mean(dim=0)*glm0.output_NL.beta) - 1)/glm0.output_NL.beta

    glm0.prepare_regularization()
    glm0.to(device)

    optimizer = torch.optim.LBFGS(glm0.parameters(),
                history_size=10,
                max_iter=10e3,
                tolerance_change=1e-9,
                line_search_fn=None,
                tolerance_grad=1e-5)

    trainer = LBFGSTrainer(
        optimizer=optimizer,
        device=device,
        max_epochs=1,
        optimize_graph=True,
        log_activations=False,
        set_grad_to_none=False,
        verbose=2)

    trainer.fit(glm0, train_data)

    llval = eval_model(glm0, val_data)
    plt.plot(llval)
    plt.axhline(0, color='k')

    ws = glm0.core[0].get_weights()
    # plot_filters_ST3D(ws[...,np.argsort(llval)])

    # #%%
    # plot_filters_ST3D(ws[...,np.where(llval>0)[0]])
    # # ws.shape
    # # %%

    NC = ws.shape[-1]
    cids = np.where(llval>0)[0]
    sx = int(np.ceil(np.sqrt(NC)))
    sy = int(np.round(np.sqrt(NC)))

    plt.figure(figsize=(10,10))
    for ii,cc in enumerate(cids):
        rf = mt.get_rf(ws, cc)

        plt.subplot(sx,sy,ii+1)
        
        xx = np.meshgrid(mt.xax, mt.yax)
        dx = rf['dx']
        dy = rf['dy']
        amp = rf['amp']
        plt.quiver(xx[0]-np.mean(xx[0]), xx[1]-np.mean(xx[1]), dx/np.max(amp), dy/np.max(amp), amp,
                    pivot='tail', scale=15, width=.008,
                    cmap=plt.cm.coolwarm)
                    
                    # ,units='width', width=.1,
                    # ,
                    # scale=20, headwidth=.002, headlength=.005)

        plt.xlim((-15,15))
        plt.ylim((-15,15))
        if ii % 3 != 0:
            plt.yticks([])
        
        if ii < 6:
            plt.xticks([])

        plt.axhline(0, color='k')
        plt.axvline(0, color='k')

    plt.savefig('../figures/MT_RF/examples_spatial_%s.pdf' %sessname)
    plt.show()

    # plot temporal kernels
    plt.figure(figsize=(10,10))
    for ii,cc in enumerate(cids):
        rf = mt.get_rf(ws, cc)

        plt.subplot(sx, sy,ii+1)
        
        plt.plot(rf['lags'], rf['tpeak'], '-o', color=plt.cm.coolwarm(np.inf))
        plt.plot(rf['lags'], rf['tmin'], '-o', color=plt.cm.coolwarm(-np.inf))
        plt.axhline(0, color='k')

        if ii % 3 != 0:
            plt.yticks([])
        
        if ii < 6:
            plt.xticks([])

        plt.axhline(0, color='k')
        
    plt.savefig('../figures/MT_RF/examples_temporal_%s.pdf' %sessname)
    plt.show()

    import seaborn as sns
    plt.rcParams.update({'font.size': 6})
    rfs = []
    tcs = []
    clist = []
    r2 = []
    for cc in cids:
        
        try:
            rf = mt.get_rf(ws, cc)
            
            dx = rf['dx']
            dy = rf['dy']
            amp = rf['amp']

            plt.figure(figsize=(2,2))
            plt.quiver(xx[0]-np.mean(xx[0]), xx[1]-np.mean(xx[1]), dx/np.max(amp), dy/np.max(amp), amp,
                            pivot='tail', scale=10, width=.01,
                            cmap=plt.cm.coolwarm)

            plt.axhline(0, color='gray', linewidth=.25)
            plt.axvline(0, color='gray', linewidth=.25)

            plt.xlabel('Azimuth (d.v.a.)')
            plt.ylabel('Elevation (d.v.a)')
            # plt.show()
            plt.savefig('../figures/example_spatial_%s_%d.pdf' %(sessname,cc))

            plt.figure(figsize=(2,2))
            plt.plot(rf['lags'], rf['tpeak']*100, '-o', color=plt.cm.coolwarm(np.inf), ms=3)
            plt.plot(rf['lags'], rf['tmin']*100, '-o', color=plt.cm.coolwarm(-np.inf), ms=3)
            plt.xlabel('Lags (ms)')
            plt.ylabel('Power (along preferred direction)')

            plt.axhline(0, color='gray')
            sns.despine(offset=0, trim=True)
            # plt.show()
            plt.savefig('../figures/example_temporal_%s_%d.pdf' %(sessname,cc))

            plt.figure(figsize=(2,2))
            tc = mt.plot_tuning_curve(cc, rf['amp'])
            r2.append(tc['r2'])
            rfs.append(rf)
            tcs.append(tc)
            clist.append(cc)

            plt.ylim(ymin=0)
            sns.despine(offset=0, trim=True)
            # plt.show()
            plt.savefig('../figures/MT_RF/example_tuning_%s_%d.pdf' %(sessname,cc))
        except:
            pass


    mu = np.mean(np.asarray(r2))

    se = np.std(np.asarray(r2)) / np.sqrt(len(r2))
    print("mean r2 = %02.3f +- %02.3f (n=%d)" % (mu, se, len(r2)))

    sessfit = {'sessname': sessname,
        'glm': glm0,
        'llval': llval,
        'cids': clist,
        'numspikes': mt.robs.sum(dim=0),
        'duration': NT/int(1/np.median(np.diff(mt.frameTime))),
        'tcs': tcs,
        'rfs': rfs}

    with open(fname, 'wb') as handle:
        pickle.dump(sessfit, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return sessfit


# %% fit them all
r2s = []
dpref = []
bw = []
num_spikes = []
num_spikes_sess = []
num_samples = []
frates = []
weights = np.empty([2,15,15,18,0])

for i in range(len(sess_list)):
    sessname = sess_list[i]
    sessfit = fit_session(sessname, dirpath=dirpath)
    mt = MTDotsDataset(sessname, dirpath)
    NT = len(mt)
    frate = int(1/np.median(np.diff(mt.frameTime)))
    num_samples.append(NT)
    frates.append(frate)
    num_spikes + list(mt.robs.sum(dim=0).numpy().astype(int))
    num_spikes_sess.append(mt.robs.sum(dim=0).numpy().astype(int))


    print("Done")
    plt.close("all")
    
    weights = np.append(weights, sessfit['glm'].core.lin0.get_weights(), axis=4)

    r = [sessfit['tcs'][cc]['r2'][0] for cc in range(len(sessfit['tcs']))]
    r2s = r2s + r
    d = [sessfit['tcs'][cc]['popt'][0] for cc in range(len(sessfit['tcs']))]
    dpref = dpref + d

    b = [sessfit['tcs'][cc]['popt'][1] for cc in range(len(sessfit['tcs']))]
    bw = bw + b

#%% Some summary
frate = 100

for i in range(len(sess_list)):
    sessname = sess_list[i]
    sessfit = fit_session(sessname, dirpath=dirpath)
    
    ngood = np.sum(sessfit['llval']>0)
    NC = len(sessfit['llval'])
    duration = num_samples[i]/frate
    print("%s %02.2f s %d/%d (%2.2f), ll=%2.3f" %(sessname, duration, ngood, NC, ngood/NC, np.mean(sessfit['llval'][sessfit['llval']>0])))

# %%

r2 = np.asarray(r2s)
ix = r2 > .5
mu = np.mean(r2)
se = np.std(r2) / np.sqrt(len(r2))
print("mean r2 = %02.3f +- %02.3f (n=%d)" % (mu, se, len(r2)))

dpref = np.asarray(dpref)
bw = np.asarray(bw)

print("mean duration = %2.2f s" %np.mean(np.asarray(num_samples)/frate))
print("min duration = %2.2f s" %np.min(np.asarray(num_samples)/frate))
print("max duration = %2.2f s" %np.max(np.asarray(num_samples)/frate))

plt.plot(dpref, bw, '.')
plt.ylim([0,20])

#%%
def wrapto360(x):
    return x%360

def rad2deg(x):
    return x/np.pi*180

plt.figure()
plt.hist(wrapto360(rad2deg(dpref[ix])), bins=np.arange(0,350,15))
plt.show()


# %%
sessfit['tcs'][0]['pcov'][0]
# %%

# %%
