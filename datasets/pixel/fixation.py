import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
from datasets.utils import downsample_time
from .utils import get_stim_list, download_set, shift_im

class FixationMultiDataset(Dataset):

    def __init__(self,
        sess_list,
        dirname,
        stimset="Train",
        requested_stims=["Gabor"],
        downsample_s: int=1,
        downsample_t: int=2,
        num_lags: int=12,
        num_lags_pre_sac: int=12,
        saccade_basis = None,
        max_fix_length: int=1000,
        download=True,
        flatten=True,
        crop_inds=None,
        min_fix_length: int=50,
        valid_eye_rad=5.2,
        add_noise=0,
        verbose=True):

        self.dirname = dirname
        self.stimset = stimset
        self.requested_stims = requested_stims
        self.downsample_s = downsample_s
        self.downsample_t = downsample_t
        self.spike_sorting = 'kilowf' # only one option for now
        self.valid_eye_rad = valid_eye_rad
        self.min_fix_length = min_fix_length
        self.flatten = flatten
        self.num_lags = num_lags
        self.num_lags_pre_sac = num_lags_pre_sac
        self.normalizing_constant = 50
        self.max_fix_length = max_fix_length
        self.saccade_basis = saccade_basis
        self.shift = None # default shift to None. To provide shifts, set outside this class. Should be a list of shift values equal to size dataset.eyepos in every way
        self.add_noise = add_noise

        if self.saccade_basis is not None:
            if type(self.saccade_basis) is np.array:
                self.saccadeB = self.saccade_basis
            else:
                if type(self.saccade_basis) is not dict or 'max_len' not in self.saccade_basis.keys():
                    self.saccade_basis['max_len'] = 40
                if type(self.saccade_basis) is not dict or 'num' not in self.saccade_basis.keys():
                    self.saccade_basis['num'] = 15
                self.saccadeB = np.maximum(1 - np.abs(np.expand_dims(np.asarray(np.arange(0,self.saccade_basis['max_len'])), axis=1) - np.arange(0,self.saccade_basis['max_len'],self.saccade_basis['max_len']/self.saccade_basis['num']))/self.saccade_basis['max_len']*self.saccade_basis['num'], 0)
        else:
            self.saccadeB = None
            
        # find valid sessions
        stim_list = get_stim_list() # list of valid sessions
        new_sess = []
        for sess in sess_list:
            if sess in stim_list.keys():
                if verbose:
                    print("Found [%s]" %sess)
                new_sess.append(sess)

        self.sess_list = new_sess # is a list of valid sessions
        self.fnames = [get_stim_list(sess) for sess in self.sess_list] # is a list of filenames

        # check if files exist. download if they don't
        for isess,fname in enumerate(self.fnames):
            fpath = os.path.join(dirname, fname)
            if not os.path.exists(fpath):
                print("File [%s] does not exist. Download set to [%s]" % (fpath, download))
                if download:
                    print("Downloading set...")
                    download_set(self.sess_list[isess], dirname)
                else:
                    print("Download is False. Exiting...")

        # open hdf5 files as a list of handles
        self.fhandles = [h5py.File(os.path.join(dirname, fname), 'r') for fname in self.fnames]

        # build index map
        self.file_index = [] # which file the fixation corresponds to
        self.stim_index = [] # which stimulus the fixation corresponds to
        self.fixation_inds = [] # the actual index into the hdf5 file for this fixation
        self.eyepos = []

        self.unit_ids_orig = []
        self.unit_id_map = []
        self.unit_ids = []
        self.num_units = []
        self.NC = 0       
        self.time_start = 0
        self.time_stop = 0

        for f, fhandle in enumerate(self.fhandles): # loop over experimental sessions
            
            # store neuron ids
            # unique_units = np.unique(fhandle['Neurons'][self.spike_sorting]['cluster'][:]).astype(int)
            unique_units = fhandle['Neurons'][self.spike_sorting]['cids'][0,:].astype(int)
            self.unit_ids_orig.append(unique_units)
            
            # map unit ids to index into the new ids
            mc = np.max(unique_units)+1
            unit_map = -1*np.ones(mc, dtype=int)
            unit_map[unique_units] = np.arange(len(unique_units))+self.NC
            self.unit_id_map.append(unit_map)

            # number of units in this session
            self.num_units.append(len(self.unit_ids_orig[-1]))

            # new unit ids
            self.unit_ids.append(np.arange(self.num_units[-1])+self.NC)
            self.NC += self.num_units[-1]

            # loop over stimuli
            for s, stim in enumerate(self.requested_stims): # loop over requested stimuli
                if stim in fhandle.keys(): # if the stimuli exist in this session
                    
                    sz = fhandle[stim]['Train']['Stim'].attrs['size']
                    self.dims = [1, int(sz[0]), int(sz[1])]

                    # get fixationss
                    labels = fhandle[stim][stimset]['labels'][:] # labeled eye positions
                    labels = labels.flatten()
                    labels[0] = 0 # force ends to be 0, so equal number onsets and offsets
                    labels[-1] = 0
                    fixations = np.diff( (labels ==  1).astype(int)) # 1 is fixation
                    fixstart = np.where(fixations==1)[0]
                    fixstop = np.where(fixations==-1)[0]

                    # offset to include lags before fixation onset
                    fixstart = fixstart-self.num_lags_pre_sac
                    fixstop = fixstop[fixstart>=0]
                    fixstart = fixstart[fixstart>=0]

                    nfix = len(fixstart)
                    if verbose:
                        print("%d fixations" %nfix)

                    # get valid indices
                    # get blocks (start, stop) of valid samples
                    blocks = fhandle[stim][stimset]['blocks'][:,:]
                    valid_inds = []
                    for bb in range(blocks.shape[1]):
                        valid_inds.append(np.arange(blocks[0,bb],
                        blocks[1,bb]))
        
                    valid_inds = np.concatenate(valid_inds).astype(int)

                    for fix_ii in range(nfix): # loop over fixations
                        
                        # get the index into the hdf5 file
                        fix_inds = np.arange(fixstart[fix_ii]+1, fixstop[fix_ii])
                        fix_inds = np.intersect1d(fix_inds, valid_inds)
                        if len(np.where(np.diff(fhandle[stim][stimset]['frameTimesOe'][0,fix_inds])>0.01)[0]) > 1:
                            if verbose:
                                print("dropped frames. skipping %d" %fix_ii)

                        if len(fix_inds) > self.max_fix_length:
                            fix_inds = fix_inds[:self.max_fix_length]

                        # check if the fixation meets our requirements to include
                        # sample eye pos
                        ppd = fhandle[stim][self.stimset]['Stim'].attrs['ppd'][0]
                        centerpix = fhandle[stim][self.stimset]['Stim'].attrs['center'][:]
                        eye_tmp = fhandle[stim][self.stimset]['eyeAtFrame'][1:3,fix_inds].T
                        eye_tmp[:,0] -= centerpix[0]
                        eye_tmp[:,1] -= centerpix[1]
                        eye_tmp/= ppd

                        if len(fix_inds) < self.min_fix_length:
                            if verbose:
                                print("fixation too short. skipping %d" %fix_ii)
                            continue
                        
                        # is the eye position outside the valid region?
                        if np.mean(np.hypot(eye_tmp[(self.num_lags_pre_sac+5):,0], eye_tmp[(self.num_lags_pre_sac+5):,1])) > self.valid_eye_rad:
                            if verbose:
                                print("eye outside valid region. skipping %d" %fix_ii)
                            continue

                        dx = np.diff(eye_tmp, axis=0)
                        vel = np.hypot(dx[:,0], dx[:,1])
                        vel[:self.num_lags_pre_sac+5] = 0
                        # find missed saccades
                        potential_saccades = np.where(vel[5:]>0.1)[0]
                        if len(potential_saccades)>0:
                            sacc_start = potential_saccades[0]
                            valid = np.arange(0, sacc_start)
                        else:
                            valid = np.arange(0, len(fix_inds))
                        
                        if len(valid)>self.min_fix_length:
                            self.eyepos.append(eye_tmp[valid,:])
                            self.fixation_inds.append(fix_inds[valid])
                            self.file_index.append(f) # which datafile does the fixation correspond to
                            self.stim_index.append(s) # which stimulus does the fixation correspond to
            
            if crop_inds is None:
                self.crop_inds = [0, self.dims[1], 0, self.dims[2]]
            else:
                self.crop_inds = [crop_inds[0], crop_inds[1], crop_inds[2], crop_inds[3]]
                self.dims[1] = crop_inds[1]-crop_inds[0]
                self.dims[2] = crop_inds[3]-crop_inds[2]

    def __getitem__(self, index):
        """
        Get item for a Fixation dataset.
        Each element in the index corresponds to a fixation. Each fixation will have a variable length.
        Concatenate fixations along the batch dimension.

        """
        stim = []
        robs = []
        dfs = []
        eyepos = []
        frames = []
        sacB = []
        fix_n = []
        

        # handle indices (can be a range, list, int, or slice). We need to convert ints, and slices into an iterable for looping
        if isinstance(index, int) or isinstance(index, np.int64):
            index = [index]
        elif isinstance(index, slice):
            index = list(range(index.start or 0, index.stop or len(self.fixation_inds), index.step or 1))
        # loop over fixations
        for ifix in index:
            fix_inds = self.fixation_inds[ifix] # indices into file for this fixation
            file = self.file_index[ifix]
            stimix = self.stim_index[ifix] # stimulus index for this fixation

            """ STIMULUS """
            # THIS is the only line that matters for sampling the stimulus if your file is already set up right
            I = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['Stim'][:,:,fix_inds]
            
            # bring the individual values into a more reasonable range (instead of [-127,127])
            I = I.astype(np.float32)/self.normalizing_constant
            
            I = torch.tensor(I, dtype=torch.float32).permute(2,0,1) # [H,W,N] -> [N,H,W]

            if self.add_noise>0:
                I += torch.randn(I.shape)*self.add_noise
            
            # append the stimulus to the list of tensors
            if self.shift is not None:
                I = shift_im(I.unsqueeze(1), self.shift[ifix])
            else:
                I = I.unsqueeze(1)
            
            I = I[...,self.crop_inds[0]:self.crop_inds[1],self.crop_inds[2]:self.crop_inds[3]]

            stim.append(I)
            fix_n.append(torch.ones(I.shape[0], dtype=torch.int64)*ifix)

            """ SPIKES """
            # NOTE: normally this would look just like the line above, but for 'Robs', but I am operating with spike times here
            # NOTE: this is MUCH slower than just indexing into the file
            frame_times = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['frameTimesOe'][0,fix_inds]
            frame_times = np.expand_dims(frame_times, axis=1)
            if self.downsample_t>1:
                frame_times = downsample_time(frame_times, self.downsample_t, flipped=False)

            frames.append(torch.tensor(frame_times, dtype=torch.float32))
            frame_times = frame_times.flatten()

            spike_inds = np.where(np.logical_and(
                self.fhandles[file]['Neurons'][self.spike_sorting]['times']>=frame_times[0],
                self.fhandles[file]['Neurons'][self.spike_sorting]['times']<=frame_times[-1]+0.01)
                )[1]

            st = self.fhandles[file]['Neurons'][self.spike_sorting]['times'][0,spike_inds]
            clu = self.fhandles[file]['Neurons'][self.spike_sorting]['cluster'][0,spike_inds].astype(int)
            # only keep spikes that are in the requested cluster ids list
            ix = np.in1d(clu, self.unit_ids_orig[file])
            st = st[ix]
            clu = clu[ix]
            # map cluster id to a unit number
            clu = self.unit_id_map[file][clu]
            
            # do the actual binning
            # robs_tmp = bin_population(st, clu, frame_times, self.unit_ids[file], maxbsize=1.2/240)
            
            # if self.downsample_t>1:
            #     robs_tmp = downsample_time(robs_tmp, self.downsample_t, flipped=False)

            robs_tmp = torch.sparse_coo_tensor( np.asarray([np.digitize(st, frame_times)-1, clu]),
                 np.ones(len(clu)), (len(frame_times), self.NC) , dtype=torch.float32)
            robs_tmp = robs_tmp.to_dense()
            robs.append(robs_tmp)

            """ DATAFILTERS """
            NCbefore = int(np.asarray(self.num_units[:file]).sum())
            NCafter = int(np.asarray(self.num_units[file+1:]).sum())
            dfs_tmp = torch.cat(
                (torch.zeros( (len(frame_times), NCbefore), dtype=torch.float32),
                torch.ones( (len(frame_times), self.num_units[file]), dtype=torch.float32),
                torch.zeros( (len(frame_times), NCafter), dtype=torch.float32)),
                dim=1)
            dfs_tmp[:self.num_lags,:] = 0 # temporal convolution will be invalid for the filter length

            # if self.downsample_t>1:
            #     dfs_tmp = downsample_time(dfs_tmp, self.downsample_t, flipped=False)

            dfs.append(dfs_tmp)

            """ EYE POSITION """
            ppd = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['Stim'].attrs['ppd'][0]
            centerpix = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['Stim'].attrs['center'][:]
            eye_tmp = self.fhandles[file][self.requested_stims[stimix]][self.stimset]['eyeAtFrame'][1:3,fix_inds].T
            eye_tmp[:,0] -= centerpix[0]
            eye_tmp[:,1] -= centerpix[1]
            eye_tmp/= ppd

            assert np.all(self.eyepos[ifix] == eye_tmp), 'eyepos does not match between object and file'
            
            if self.downsample_t>1:
                eye_tmp = downsample_time(eye_tmp, self.downsample_t, flipped=False)

            eyepos.append(torch.tensor(eye_tmp, dtype=torch.float32))

            """ SACCADES (on basis) """
            if self.saccadeB is not None:
                fix_len = len(frame_times)
                sacB_len = self.saccadeB.shape[0]
                if fix_len < sacB_len:
                    sacc_tmp = torch.tensor(self.saccadeB[:fix_len,:], dtype=torch.float32)
                else:
                    sacc_tmp = torch.cat( (torch.tensor(self.saccadeB, dtype=torch.float32),
                        torch.zeros( (fix_len-sacB_len, self.saccadeB.shape[1]), dtype=torch.float32)
                        ), dim=0)
                sacB.append(sacc_tmp)

        # concatenate along batch dimension
        stim = torch.cat(stim, dim=0)
        fix_n = torch.cat(fix_n, dim=0)
        eyepos = torch.cat(eyepos, dim=0)
        robs = torch.cat(robs, dim=0)
        dfs = torch.cat(dfs, dim=0)
        frames = torch.cat(frames, dim=0)

        if self.flatten:
            stim = torch.flatten(stim, start_dim=1)
        
        sample = {'stim': stim, 'robs': robs, 'dfs': dfs, 'eyepos': eyepos, 'frame_times': frames, 'fix_n': fix_n}

        if self.saccadeB is not None:
            sample['saccade'] = torch.cat(sacB, dim=0)

        return sample

    def __len__(self):
        return len(self.fixation_inds)

    def get_stim_indices(self, stim_name='Gabor'):
        if isinstance(stim_name, str):
            stim_name = [stim_name]
        stim_id = [i for i,s in enumerate(self.requested_stims) if s in stim_name]

        indices = [i for i,s in enumerate(self.stim_index) if s in stim_id]
        
        return indices
    
    def expand_shift(self, fix_shift, fix_inds=None):
        if fix_inds is None:
            assert fix_shift.shape[0]==len(self), 'fix_shift not equal number of fixations. Pass in fix_inds as well.'
            fix_inds = np.arange(len(self)) # index into all fixations
        
        new_shift = []
        for i, fix in enumerate(fix_inds):
            new_shift.append(fix_shift[i].repeat(len(self.fixation_inds[fix]), 1))
        
        new_shift = torch.cat(new_shift, dim=0)
        return new_shift

    def plot_shifter(self, shifter, valid_eye_rad=5.2, ngrid = 100):
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

        return shift

    def get_shifters(self, plot=False):

        shifters = {}
        for sess in self.sess_list:
            sfname = [f for f in os.listdir(self.dirname) if 'shifter_' + sess in f]
                
            if len(sfname) == 0:
                from datasets.mitchell.pixel.utils import download_shifter
                download_shifter(self.sess_list[0], self.dirname)
            else:
                print("Shifter exists")
                import pickle
                fname = os.path.join(self.dirname, sfname[0])
                shifter_res = pickle.load(open(fname, "rb"))
                shifter = shifter_res['shifters'][np.argmin(shifter_res['vallos'])]

            if plot:
                _ = self.plot_shifter(shifter)
            
            shifters[sess] = shifter

        return shifters
    

    def shift_stim(self, im, shift, unflatten=False):
        """
        apply shifter to translate stimulus as a function of the eye position
        """
        import torch.nn.functional as F
        import torch
        affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
        sz = [im.shape[0]] + self.dims

        if len(im.shape)==2:
            unflatten = True
            im = im.reshape(sz)

        aff = torch.tensor([[1,0,0],[0,1,0]])

        affine_trans = shift[:,:,None]+aff[None,:,:]
        affine_trans[:,0,0] = 1
        affine_trans[:,0,1] = 0
        affine_trans[:,1,0] = 0
        affine_trans[:,1,1] = 1

        n = im.shape[0]
        grid = F.affine_grid(affine_trans, torch.Size((n, 1, sz[-2], sz[-1])), align_corners=False)

        im2 = F.grid_sample(im, grid, align_corners=False)

        if unflatten:
            torch.flatten(im2, start_dim=1)

        return im2