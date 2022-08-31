import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import h5py
from tqdm import tqdm
import os
from datasets.utils import ensure_dir, reporthook

""" Available Datasets:
1. FixationMultiDataset - generates fixations from multiple stimulus classes 
                        and experimental sessions. no time-embedding
2. PixelDataset - time-embedded 2D free-viewing movies and spike trains
"""
def get_stim_list(id=None, verbose=False):

    stim_list = {
            '20191119': 'logan_20191119_-20_-10_50_60_0_19_0_1.hdf5',
            '20191120a':'logan_20191120a_-20_-10_50_60_0_19_0_1.hdf5',
            '20191121': 'logan_20191121_-20_-20_50_50_0_19_0_1.hdf5',
            '20191122': 'logan_20191122_-20_-10_50_60_0_19_0_1.hdf5',
            '20191205': 'logan_20191205_-20_-10_50_60_0_19_0_1.hdf5',
            '20191206': 'logan_20191206_-20_-10_50_60_0_19_0_1.hdf5',
            '20191231': 'logan_20191231_-20_-10_50_60_0_19_0_1.hdf5',
            '20200304': 'logan_20200304_-20_-10_50_60_0_19_0_1.hdf5',
            '20220216': 'allen_20220216_-60_-50_10_20_0_19_0_1.hdf5',
            '20220601': 'allen_20220601_-80_-50_10_50_1_0_1_19_0_1.hdf5',
            '20220601fix': 'allen_20220601_-80_-50_10_50_0_0_0_19_0_1.hdf5',
            '20220601fixc': 'allen_20220601_-80_-50_10_50_1_0_0_19_0_1.hdf5',
            '20220610': 'allen_20220610_-80_-50_10_50_1_0_1_19_0_1.hdf5',
            '20220610fix': 'allen_20220610_-80_-50_10_50_0_0_0_19_0_1.hdf5',
            '20220610fixc': 'allen_20220610_-80_-50_10_50_1_0_0_19_0_1.hdf5',
        }

    if id is None:
        for str in list(stim_list.keys()):
            if verbose:
                print(str)
        return stim_list

    if id not in stim_list.keys():
        raise ValueError('Stimulus not found')
    
    return stim_list[id]

def get_stim_url(id):
    urlpath = {
            '20191119': 'https://www.dropbox.com/s/xxaat202j20kriy/logan_20191119_-20_-10_50_60_0_19_0_1.hdf5?dl=1',
            '20191231':'https://www.dropbox.com/s/ulpcjfb48c6dyyf/logan_20191231_-20_-10_50_60_0_19_0_1.hdf5?dl=1',
            '20200304': 'https://www.dropbox.com/s/5tj5m2rp0wht8z2/logan_20200304_-20_-10_50_60_0_19_0_1.hdf5?dl=1',
            '20220601': 'https://www.dropbox.com/s/v85m0b9kzwiowqm/allen_20220601_-80_-50_10_50_1_0_19_0_1.hdf5?dl=1'
        }
    
    if id not in urlpath.keys():
        raise ValueError('Stimulus URL not found')
    
    return urlpath[id]

def get_shifter_url(id):
    urlpath = {
            '20191119': 'https://www.dropbox.com/s/dd05gxt8l8hmw3o/shifter_20191119_kilowf.p?dl=1',
            '20191120a': 'https://www.dropbox.com/s/h4elcp46le5tet0/shifter_20191120a_kilowf.p?dl=1',
            '20191121': 'https://www.dropbox.com/s/rfzefex8diu5ts5/shifter_20191121_kilowf.p?dl=1',
            '20191122': 'https://www.dropbox.com/s/2me7yauvpprnv0b/shifter_20191122_kilowf.p?dl=1',
            '20191205': 'https://www.dropbox.com/s/r56wt4rfozmjiy8/shifter_20191205_kilowf.p?dl=1',
            '20191206': 'https://www.dropbox.com/s/qec8cats077bx8c/shifter_20191206_kilowf.p?dl=1',
            '20200304': 'https://www.dropbox.com/s/t0j8k55a8jexgt4/shifter_20210304_kilowf.p?dl=1',
        }
    
    if id not in urlpath.keys():
        raise ValueError('Stimulus URL not found')
    
    return urlpath[id]


def download_set(sessname, fpath):
    
    ensure_dir(fpath)

    # Download the data set
    url = get_stim_url(sessname)
    fout = os.path.join(fpath, get_stim_list(sessname))
    print("Downloading...") 
    import urllib
    urllib.request.urlretrieve(url, fout, reporthook)
    print("Done")

def download_shifter(sessname, fpath):
    
    ensure_dir(fpath)

    # Download the data set
    url = get_shifter_url(sessname)
    fout = os.path.join(fpath, 'shifter_' + sessname + '_kilowf.p')
    print("Downloading...") 
    import urllib
    urllib.request.urlretrieve(url, fout, reporthook)
    print("Done")


def shift_im(im, shift, mode='nearest', upsample=1):
        """
        apply shifter to translate stimulus as a function of the eye position
        im = N x C x H x W (torch.float32)
        shift = N x 2 (torch.float32), shift along W and H, respectively
        """
        import torch.nn.functional as F
        import torch
        affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        sz = im.shape

        aff = torch.tensor([[1,0,0],[0,1,0]])

        affine_trans = shift[:,:,None]+aff[None,:,:]
        affine_trans[:,0,0] = 1
        affine_trans[:,0,1] = 0
        affine_trans[:,1,0] = 0
        affine_trans[:,1,1] = 1

        n = im.shape[0]
        grid = F.affine_grid(affine_trans, torch.Size((n, 1, sz[2], sz[3])), align_corners=False)

        if upsample > 1:
            upsamp = torch.nn.UpsamplingNearest2d(scale_factor=upsample)
            im = upsamp(im)

        im2 = F.grid_sample(im, grid, align_corners=False, mode=mode)

        return im2.detach()

def plot_shifter(shifter, valid_eye_rad=5.2, ngrid = 100, title=None):
        import matplotlib.pyplot as plt
        xx,yy = np.meshgrid(np.linspace(-valid_eye_rad, valid_eye_rad,ngrid),np.linspace(-valid_eye_rad, valid_eye_rad,ngrid))
        xgrid = torch.tensor( xx.astype('float32').reshape( (-1,1)))
        ygrid = torch.tensor( yy.astype('float32').reshape( (-1,1)))

        inputs = torch.cat( (xgrid,ygrid), dim=1)

        xyshift = shifter(inputs).detach().numpy()

        xyshift/=valid_eye_rad/60 # conver to arcmin
        vmin = np.min(xyshift)
        vmax = np.max(xyshift)

        shift = [xyshift[:,0].reshape((ngrid,ngrid))]
        shift.append(xyshift[:,1].reshape((ngrid,ngrid))) 
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.imshow(shift[0], extent=(-valid_eye_rad,valid_eye_rad,-valid_eye_rad,valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(shift[1], extent=(-valid_eye_rad,valid_eye_rad,-valid_eye_rad,valid_eye_rad), interpolation=None, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()
        if title is not None:
            plt.suptitle(title)

        return shift