
#%%
import matplotlib.pyplot as plt
import numpy as np
import h5py

#%%
fname = '/Users/jake/Dropbox/Datasets/Mitchell/stim_movies/logan_20200304_-20_-10_50_60_0_19_0_1.hdf5' 
# open h5 file
f = h5py.File(fname, 'r')

eyeX = f['BackImage']['Train']['eyeAtFrame'][1,:]/3
eyeY = f['BackImage']['Train']['eyeAtFrame'][2,:]/3

ppd = 40 # pixels per degree

plt.figure()
plt.plot(eyeX/ppd)
f.close()

#%%

eccentricity = 1

w = 10*ppd
h = 10*ppd

# screen coordinates
sxi,syi = np.meshgrid(np.arange(0,w), np.arange(0,h))

ndims = ppd
xax = np.linspace(-3.5,3.5,ndims)
xi,yi = np.meshgrid(xax,xax)

def cosd(theta):
    return np.cos(theta/180*np.pi)/np.pi*180

def sind(theta):
    return np.sin(theta/180*np.pi)/np.pi*180

def gabor(xi,yi,mu=(0,0), sigma=(1,1), freq=1, ori=45, phase=0):

    grid = cosd(ori)*xi + sind(ori)*yi
    carrier = cosd(freq*grid + phase)
    gauss = np.exp(  -.5 * ((xi - mu[0])**2/sigma[0]**2 + (yi - mu[1])**2/sigma[1]**2 ) )
    return carrier*gauss


rf = gabor(xi,yi, freq=4, sigma=(.5,.5))
plt.figure()
plt.imshow(rf)

#%% helper functions for running gaze-contingent processing
def interp_place(Ibig, Ismall, x, y):
    # place RF in a location in a bigger image
    ws,hs = Ismall.shape
    wb,hb = Ibig.shape
    # np.meshgrid()
    xax = np.arange(-ws//2, ws//2)
    yax = np.arange(-hs//2, hs//2)
    xi,yi = np.meshgrid(xax, yax)

    x0 = np.floor(x).astype(int)
    x1 = np.ceil(x).astype(int)
    y0 = np.floor(y).astype(int)
    y1 = np.ceil(y).astype(int)

    xd = x-x0, x1-x
    yd = y-y0, y1-y

    wts = np.array(  ( (xd[0] + yd[0], xd[0] + yd[1]) ,
      (xd[1] + yd[0], xd[1] + yd[1]) ) ) / 4
    
    if np.sum(wts)==0:
        wts[0,0] = 1.0

    x0 = xi + x0
    x1 = xi + x1
    y0 = yi + y0
    y1 = yi + y1

    x0 = np.minimum(np.maximum(0, x0), wb-1)
    x1 = np.minimum(np.maximum(0, x1), wb-1)
    y0 = np.minimum(np.maximum(0, y0), hb-1)
    y1 = np.minimum(np.maximum(0, y1), hb-1)

    Ibig[x0,y0] += wts[0,0]*Ismall
    Ibig[x0,y1] += wts[0,1]*Ismall
    Ibig[x1,y0] += wts[1,0]*Ismall
    Ibig[x1,y1] += wts[1,1]*Ismall
    # return x0,x1,y0,y1

    return Ibig

def interp_get(Ibig, Ismalldims, x, y):
    # get small inset from a float location in a bigger image
    ws,hs = Ismalldims
    wb,hb = Ibig.shape
    # np.meshgrid()
    xax = np.arange(-ws//2, ws//2)
    yax = np.arange(-hs//2, hs//2)
    xi,yi = np.meshgrid(xax, yax)

    x0 = np.floor(x).astype(int)
    x1 = np.ceil(x).astype(int)
    y0 = np.floor(y).astype(int)
    y1 = np.ceil(y).astype(int)

    xd = x-x0, x1-x
    yd = y-y0, y1-y

    wts = np.array(  ( (xd[0] + yd[0], xd[0] + yd[1]) ,
      (xd[1] + yd[0], xd[1] + yd[1]) ) ) / 4

    if np.sum(wts)==0:
        wts[0,0] = 1.0

    x0 = xi + x0
    x1 = xi + x1
    y0 = yi + y0
    y1 = yi + y1

    x0 = np.minimum(np.maximum(0, x0), wb-1)
    x1 = np.minimum(np.maximum(0, x1), wb-1)
    y0 = np.minimum(np.maximum(0, y0), hb-1)
    y1 = np.minimum(np.maximum(0, y1), hb-1)

    Iout = np.zeros(Ismalldims)
    Iout += wts[0,0]*Ibig[x0,y0]
    Iout += wts[0,1]*Ibig[x0,y1]
    Iout += wts[1,0]*Ibig[x1,y0]
    Iout += wts[1,1]*Ibig[x1,y1]

    return Iout


#%%
plt.figure()
RFscreen = np.zeros((w,h))
plt.imshow(interp_place(RFscreen, rf, 200.2,50.5), cmap='gray', vmin=-50, vmax=50)
plt.show()

# %% get eye trace

plt.figure()

# stimulate eye trace
NT = 10000
NT = np.minimum(NT, len(eyeX))
print("Generating %d frames of a stimulus" %NT)

xeye = eyeX[:NT]
yeye = eyeY[:NT]
plt.plot(xeye)
plt.plot(yeye)


# generate noise stimulus
seed = 1234
np.random.seed(seed)
stim = np.random.randn(NT, w, h)

print("Done")
#%% get animation blank screen
from matplotlib import animation

frame = 0
RFscreen = np.zeros((w,h))

Im = interp_place(RFscreen, rf, xeye[frame],yeye[frame])

fig = plt.figure()
im = plt.imshow(Im, interpolation='none', cmap='gray', vmin=-50, vmax=50)
plt.axis("off")

def animate(i):
    RFscreen = np.zeros((w,h))
    Im = interp_place(RFscreen, rf, xeye[i],yeye[i])
    im.set_data(Im)
    return im

n = 500
anim = animation.FuncAnimation(
    fig, animate, frames=np.arange(1,n),
    repeat=False)


mywriter = animation.FFMpegWriter(fps=30)
anim.save('movie_blank' + str(1) + '.gif', writer=mywriter)


#%%
#%% get animation blank screen
from matplotlib import animation

RFscreen = stim[0,:,:].squeeze()*50

Im = interp_place(RFscreen, rf, xeye[frame],yeye[frame])

fig = plt.figure()
im = plt.imshow(Im, interpolation='none', cmap='gray', vmin=-50, vmax=50)
plt.axis("off")

def animate(i):
    RFscreen = np.zeros((w,h))
    Im = interp_place(RFscreen, rf, xeye[i],yeye[i])
    im.set_data(Im+stim[i,:,:].squeeze()*50)
    return im

n = 500
anim = animation.FuncAnimation(
    fig, animate, frames=np.arange(1,n),
    repeat=False)


mywriter = animation.FFMpegWriter(fps=30)
anim.save('movie_noise' + str(1) + '.gif', writer=mywriter)


#%%
plt.figure()
frame += 1
# RFscreen = stim[frame,:,:].squeeze()
# plt.imshow(I, interpolation='none', cmap='gray')
RFscreen = np.zeros((w,h))
plt.imshow(interp_place(RFscreen, rf, xeye[frame],yeye[frame]), cmap='gray', vmin=-50, vmax=50)
# plt.show()


#%% get generator
generator = np.zeros(NT)

for frame in range(NT):

    # get the stimulus in the RF location
    stiminrf = interp_get(stim[frame,:,:].squeeze(), [ndims,ndims], xeye[frame], yeye[frame])
    generator[frame] = stiminrf.flatten()@rf.flatten()


# %% generate spike rate from generator signal
rate = np.maximum(0, generator/500)**2 # nonlinearity is half-squaring, adjust scale to a reasonable rate
plt.figure()
plt.plot( rate )

# spikecnt = np.random.poisson(rate)
spikecnt = rate


# %%



def get_sta_with_eye_tracking(noisesigma=0, calibration=1):
    sta = 0
    x0 = np.mean(xeye)
    y0 = np.mean(yeye)
    for frame in range(NT):
        stiminrf = interp_get(stim[frame,:,:].squeeze(), [ndims,ndims], calibration*(xeye[frame]-x0)+x0+np.random.randn()*noisesigma, calibration*(yeye[frame]-y0) + y0 + np.random.randn()*noisesigma)
        sta += stiminrf*spikecnt[frame]
    
    return sta / np.sum(spikecnt)
 
#%%
eyetracknoise = [0] # ,1,2,4,8
plt.figure(figsize=(10,5))
plt.subplot(1,len(eyetracknoise)+1, 1)
plt.imshow(rf, cmap='gray', vmin=-np.max(rf), vmax=np.max(rf))
plt.title('True RF')
plt.axis("off")
for i,noise in enumerate(eyetracknoise):
    plt.subplot(1, len(eyetracknoise)+1, i+2)
    sta = get_sta_with_eye_tracking(noise)
    plt.imshow(sta, cmap=plt.cm.gray, vmin=-np.max(sta), vmax=np.max(sta))
    plt.axis("off")
    plt.title("Measured")

plt.savefig('eyetrack_perfect_calibration.pdf')
plt.show()

# %%
eyetracknoise = [0,1,2,4,8]
calibration_noise = [1, 1.025, 1.05, 1.1, 1.15]
plt.figure(figsize=(10,10))
sta = get_sta_with_eye_tracking(noisesigma=0, calibration=1)

vmax = np.max(np.abs(sta))
# vmin = np.min(sta)
vmin = -vmax

for i,gain in enumerate(calibration_noise):
    for j,noise in enumerate(eyetracknoise):
        # print(i,j)
        
        sta = get_sta_with_eye_tracking(noisesigma=noise, calibration=gain)

        plt.subplot(len(eyetracknoise), len(calibration_noise), j*len(eyetracknoise)+i+1)
        plt.imshow(sta, vmin=vmin, vmax=vmax, cmap=plt.cm.gray)
        # plt.imshow(sta, cmap=plt.cm.gray, vmin=)
        plt.axis('off')

plt.savefig('eyetrack_noise_calibration.pdf')
plt.show()
# %%
plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot( (xeye + np.random.randn(NT)*1)/ppd, color=[0.5,0.5,0.5])
plt.plot(xeye/ppd, color='k')
plt.xlim((1300,1800))
plt.ylim((4.45,5.65))
plt.axis("off")

plt.subplot(2,1,2)
plt.plot(xeye/ppd, color='k')
plt.plot( (1.1*(xeye - np.mean(xeye)) + np.mean(xeye))/ppd, color=[0.5,0.5,0.5])
plt.xlim((1300,1800))
plt.ylim((4.45,5.65))
plt.axis("off")
plt.savefig('sampletraces.pdf')
plt.show()

# %%

plt.figure(figsize=(5,2))
plt.subplot(1,2,1)
plt.imshow(rf, cmap=plt.cm.gray, vmin=-1, vmax=2)
plt.axis('off')
plt.subplot(1,2,2)
plt.plot(np.linspace(0,1,100), np.linspace(0,1,100)**2, color='k')
plt.axhline(0, color='k')
plt.axis('off')
plt.savefig('rf.pdf')

# %%

plt.figure()
I = stim[0,:,:].squeeze()
# I[np.where(np.abs(I)<3)]
# plt.imshow(I, cmap=plt.cm.gray)
# %%
