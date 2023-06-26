function fname = make_stimulus_file_for_py(Exp, S, varargin)
% make_stimulus_file_for_py(Exp, S, varargin)

if isfield(S, 'spikeSorting')
    spike_sorting = S.spikeSorting;
else
    spike_sorting = 'none';
end

ip = inputParser();
ip.addParameter('stimlist', {'Dots', 'Gabor', 'Grating', 'FixRsvpStim', 'BackImage'})
ip.addParameter('overwrite', false)
ip.addParameter('includeProbe', true)
ip.addParameter('GazeContingent', true)
ip.addParameter('EyeCorrection', [])
ip.addParameter('EyeSmoothing', 41)
ip.addParameter('EyeSmoothingOrder', 3)
ip.addParameter('usePTBdraw', true)
ip.addParameter('validTrials', [])
ip.parse(varargin{:});
stimlist = ip.Results.stimlist;
eyesmoothing = ip.Results.EyeSmoothing;
overwrite = ip.Results.overwrite;

for istim = 1:numel(stimlist)
    
    stimset = stimlist{istim};
    validTrials = io.getValidTrials(Exp, stimset);
    if ~isempty(ip.Results.validTrials)
        validTrials = intersect(validTrials, ip.Results.validTrials);
    end

    if numel(validTrials) < 2
        fprintf('make_stimulus_file: No Trials. Skipping [%s]\n', stimset)
        continue
    end
    options = {'stimulus', stimset, ...
        'debug', false, ...
        'testmode', true, ...
        'eyesmooth', eyesmoothing, ... % bins
        'eyesmoothorder', ip.Results.EyeSmoothingOrder, ...
        'includeProbe', ip.Results.includeProbe, ...
        'correctEyePos', ip.Results.EyeCorrection, ...
        'nonlinearEyeCorrection', false, ...
        'usePTBdraw', ip.Results.usePTBdraw, ...
        'GazeContingent', ip.Results.GazeContingent, ...
        'validTrials', validTrials, ...
        'overwrite', overwrite};
    
    % Test set
    options{find(strcmp(options, 'testmode')) + 1} = true;
    fname = io.dataGenerateHdf5(Exp, S, options{:});
    
    h5writeatt(fname, ['/' stimset '/Test/Stim'], 'frate', Exp.S.frameRate)
    h5writeatt(fname, ['/' stimset '/Test/Stim'], 'center', Exp.S.centerPix)
    h5writeatt(fname, ['/' stimset '/Test/Stim'], 'viewdist', Exp.S.screenDistance)
    
    % Training/Validation set
    options{find(strcmp(options, 'testmode')) + 1} = false;
    fname = io.dataGenerateHdf5(Exp, S, options{:});
    
    
    h5writeatt(fname, ['/' stimset '/Train/Stim'], 'frate', Exp.S.frameRate)
    h5writeatt(fname, ['/' stimset '/Train/Stim'], 'center', Exp.S.centerPix)
    h5writeatt(fname, ['/' stimset '/Train/Stim'], 'viewdist', Exp.S.screenDistance)
    
end

% TODO: include CSD depths / burstiness index / different spike sorting
    
% NOTE: Matlab's hdf5 utilities are really dumb and return errors if they
% can't find an address or if it already exists. There seems to be no way
% to check if it exists in one line and I'm not writing something involved
% to handle it. Instead, we wrap each of these additions in a try statement

% Add Neuron meta data
try %#ok<*TRYNC>
    h5create(fname,'/Neurons/cgs', size(Exp.osp.cgs))
    h5write(fname, '/Neurons/cgs', Exp.osp.cgs)
end

if ~strcmp(spike_sorting, 'none')
    io.h5addSpikesorting(Exp, fname, spike_sorting)
end


