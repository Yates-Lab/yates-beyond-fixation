

#%%
import h5py
import numpy as np
import scipy.io as sio
#%%

fname = '/Users/jake/Dropbox/Datasets/Mitchell/hires/allen_20220610.mat'

f = h5py.File(fname, 'r')


# %%
trials = f['D'][:]
# trials[0].visititems(lambda n,o:print(n, o))
# f['D'][0][0].keys()
# %%
ref = f['D'][0][0]
trial = f[ref]
trial.visititems(lambda n,o:print(n, o))
# %%
fname = '/Users/jake/Dropbox/Datasets/Mitchell/stim_movies/logan_20200304_-31_-2_49_78_1_1_1_19_0_0.hdf5'

f = h5py.File(fname, 'r')

# %%
f.keys()
# %%
f['BackImage'].visititems(lambda n,o:print(n, o))
# %%

#%%
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import time
sz = 100
n = 6000
# xy = np.random.randint(0, sz, (n,2))
x = np.random.randint(0, sz, n)
y = np.random.randint(0, sz, n)

t0 = time.time()
F = coo_matrix((np.ones(n), (x,y)), shape=(sz, sz))
F = F.toarray()
# F = (F/F.max()*255).astype(np.uint8)
print(time.time() - t0)
plt.imshow(F, interpolation='none')
# make a sparse matrix of size (sz, sz), with inputs x,y,v
# then convert to a dense matrix
# %%
