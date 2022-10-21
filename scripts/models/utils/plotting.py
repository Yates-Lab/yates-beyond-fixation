import numpy as np
import matplotlib.pyplot as plt

def plot_stas(stas, show_zero=True, plot=True, thresh=None):
    
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
        plt.figure(figsize=(sx*3,sy*2))

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
        
        try:
            if plot:
                plt.subplot(sx,sy, cc*2 + 2)
            i,j=np.where(w[bestlag,:,:]==np.max(w[bestlag,:,:]))
            t1 = stas[:,i[0],j[0],cc]
            # t1 = w[:,i[0], j[0]]
            if plot:
                plt.plot(t1, '-ob')
            i,j=np.where(w[bestlag,:,:]==np.min(w[bestlag,:,:]))
            # t2 = w[:,i[0], j[0]]
            t2 = stas[:,i[0],j[0],cc]
            if plot:
                plt.plot(t2, '-or')
                if show_zero:
                    plt.axhline(0, color='k')
                    if thresh is not None:
                        plt.axhline(thresh[cc],color='k', ls='--')
                
        except:
            pass
    
    return mu, blag.astype(int)