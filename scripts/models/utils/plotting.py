import numpy as np
import matplotlib.pyplot as plt

def plot_stas(stas, show_zero=True, plot=True, thresh=None, title=None):
    
    NC = stas.shape[-1]
    num_lags= stas.shape[0]

    sx = int(np.ceil(np.sqrt(NC*2)))
    sy = int(np.round(np.sqrt(NC*2)))
    mod2 = sy % 2
    sy += mod2
    sx -= mod2
    mu = np.zeros((NC,2))
    amp = np.zeros(NC)
    blag = np.zeros(NC)

    if plot:
        fig = plt.figure(figsize=(sx*3,sy*2))
    else:
        fig = None

    for cc in range(NC):
        w = stas[:,:,:,cc]

        wt = np.std(w, axis=0)
        wt /= np.max(np.abs(wt)) # normalize for numerical stability
        # softmax
        wt = wt**10
        wt /= np.sum(wt)
        sz = wt.shape
        xx,yy = np.meshgrid(np.linspace(-1, 1, sz[1]), np.linspace(1, -1, sz[0]))

        mu[cc,0] = np.minimum(np.maximum(np.sum(xx*wt), -.5), .5) # center of mass after softmax
        mu[cc,1] = np.minimum(np.maximum(np.sum(yy*wt), -.5), .5) # center of mass after softmax

        w = (w -np.mean(w) )/ np.std(w)

        bestlag = np.argmax(np.std(w.reshape( (num_lags, -1)), axis=1))
        blag[cc] = bestlag
        
        v = np.max(np.abs(w))
        amp[cc] = np.std(w[bestlag,:,:].flatten())

        if plot:
            plt.subplot(sx,sy, cc*2 + 1)
            plt.imshow(w[bestlag,:,:], aspect='auto', interpolation=None, vmin=-v, vmax=v, cmap="coolwarm_r", extent=(-1,1,-1,1))
            plt.title(cc)
        
        if plot:
            try:
                plt.subplot(sx,sy, cc*2 + 2)
                i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
                t1 = stas[:,i[0],j[0],cc]
                plt.plot(t1, '-ob')
                i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
                t2 = stas[:,i[0],j[0],cc]
                plt.plot(t2, '-or')
                if show_zero:
                    plt.axhline(0, color='k')
                    if thresh is not None:
                        plt.axhline(thresh[cc],color='k', ls='--')
            except:
                pass
        
        if plot and title is not None:
            plt.suptitle(title)
    
    return mu, blag.astype(int), fig

def plot_layer(layer, ind=None):
    
    if layer.filter_dims[-1] > 1:
        ws = layer.get_weights()
        if ind is not None: 
            ws = ws[..., ind]
        ws = np.transpose(ws, (2,0,1,3))
        layer.plot_filters()
    else:
        ws = layer.get_weights()
        if ind is not None: 
            ws = ws[..., ind]

        nout = ws.shape[-1]
        nin = ws.shape[0]
        plt.figure(figsize=(10, 10))

        for i in range(nin):
            for j in range(nout):
                plt.subplot(nin, nout, i*nout + j + 1)
                plt.imshow(ws[i, :,:, j], aspect='auto', interpolation='none')
                plt.axis("off")

def plot_dense_readout(layer):
    ws = layer.get_weights()
    n = ws.shape[-1]
    sx = int(np.ceil(np.sqrt(n)))
    sy = int(np.ceil(np.sqrt(n)))
    plt.figure(figsize=(sx*2, sy*2))
    for cc in range(n):
        plt.subplot(sx, sy, cc + 1)
        v = np.max(np.abs(ws[:,:,cc]))
        plt.imshow(ws[:,:,cc], interpolation='none', cmap=plt.cm.coolwarm, vmin=-v, vmax=v)

def plot_model(model):

    for layer in model.core:
        plot_layer(layer)
        plt.show()

    if hasattr(model, 'offsets'):
        for i,layer in enumerate(model.offsets):
            _ = plt.plot(layer.get_weights())
            plt.title("Offset {}".format(model.offsetstims[i]))
            plt.show()
    
    if hasattr(model, 'gains'):
        for i,layer in enumerate(model.gains):
            _ = plt.plot(layer.get_weights())
            plt.title("Gain {}".format(model.gainstims[i]))
            plt.show()

    if hasattr(model.readout, 'mu'):
        plt.imshow(model.readout.get_weights())
    else:
        plot_dense_readout(model.readout.space)
        plt.show()
        plt.imshow(model.readout.feature.get_weights())
        plt.xlabel("Neuron ID")
        plt.ylabel("Feature ID")