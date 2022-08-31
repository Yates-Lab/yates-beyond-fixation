function [stim, spks, opts] = preprocess_grating_subspace_data(Exp, varargin)
% Preprocess data for analysis
% Bin the stimulus / spikes and keep track of all parameters used
% Inputs:
%   D (data struct) - has fields corresponding to the stimulus of interest
%                     and spikes
%   Optional Arguments:
%       'fs_stim'           - sampling resolution to bin the stimulus at
%       'up_samp_fac',      - multiplier to bin spikes at higher resolution
%       'num_lags_stim'     - number of lags to include for a stimulus filter
%       'num_lags_sac_post' - number of lags after a saccade (in spike
%                             sampling resolution)
%       'num_lags_sac_pre'  - number of lags before a saccade (in spike
%                             resolution)
%       'true_zero'         - The origin of the fourier domain is currently
%                             one of the hartley stimulus options. Zero it
%                             out because it is really just a gray screen.
%       'trial_inds'        - list of trials to analyze
% Outputs:
%   stim [cell array]   - (T x m) stimuli: stimulus, saccade_onsets, saccade_offsets, eye velocity
%   spks,[T x n]        - binned spike counts
%   opts [struct]       - information about the preprocessing
%
% Example:
%   [stim, spks, opts, params_stim] = io.preprocess_grating_subspace_data(D, 'stim_field', 'hartleyFF', 'fs_stim', 120);

% jly 2020
ip = inputParser();
ip.KeepUnmatched = true;
ip.addParameter('fs_stim', [])      % stimulus binning rate (per second)
ip.addParameter('up_samp_fac', 1)    % spikes are sampled at higher temporal res?
ip.addParameter('num_lags_stim', 20) % number of lags for stimulus filter
ip.addParameter('num_lags_sac_post', 50)  % number of time lags to capture saccade effect
ip.addParameter('num_lags_sac_pre', 50)  % number of time lags to capture saccade effect
ip.addParameter('stim_field', 'Grating')
ip.addParameter('trial_inds', [])
ip.addParameter('eye_pos',[])
ip.addParameter('force_hartley', false)
ip.addParameter('validTrials', [])
ip.parse(varargin{:})

stimField = ip.Results.stim_field;

opts = ip.Results;
if isempty(ip.Results.fs_stim)
    opts.fs_stim = Exp.S.frameRate;
end

opts.fs_spikes = opts.fs_stim;
opts.up_samp_fac = 1;

if isempty(ip.Results.validTrials)
    validTrials = io.getValidTrials(Exp, stimField);
else
    validTrials = ip.Results.validTrials;
end

eyeDat = Exp.vpx.smo(:,1:3); % eye position
% convert time to ephys units
eyeDat(:,1) = Exp.vpx2ephys(eyeDat(:,1));

% edge-case: sometimes high-res eye data wasn't collected for all trials
% exclude trials without high-res eye data
tstarts = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
goodix = tstarts > eyeDat(1,1) & tstarts < eyeDat(end,1);

validTrials = validTrials(goodix);

if numel(validTrials) < 2
    stim = [];
    spks = [];
    return
end

frameTime = cellfun(@(x) Exp.ptb2Ephys(x.PR.NoiseHistory(:,1)), Exp.D(validTrials), 'uni', 0);
ori = cellfun(@(x) x.PR.NoiseHistory(:,2), Exp.D(validTrials), 'uni', 0);
cpd = cellfun(@(x) x.PR.NoiseHistory(:,3), Exp.D(validTrials), 'uni', 0);

trialDuration = cellfun(@numel , frameTime);
trialStartFrameNum = 1 + cumsum([0; trialDuration(1:end-1)]);
trialEndFrameNum = trialDuration + trialStartFrameNum - 1;

% frozenSequence starts, duration
hasFrozen = cellfun(@(x) isfield(x.PR, 'frozenSequence'), Exp.D(validTrials));
frozenTrials = cellfun(@(x) x.PR.frozenSequence, Exp.D(validTrials(hasFrozen)));
if ~any(frozenTrials)
    frozen_seq_starts = [];
    frozen_seq_dur = 0;
else
    frozen_seq_dur = cellfun(@(x) x.PR.frozenSequenceLength, Exp.D(validTrials(frozenTrials)));
    frozen_seq_dur = min(frozen_seq_dur);
    frozen_seq_starts = [];
    for iTrial = find(frozenTrials(:))'
        numRepeats = floor(trialDuration(iTrial)/frozen_seq_dur);
        trial_seq_starts = trialStartFrameNum(iTrial) + (0:(numRepeats-1))*frozen_seq_dur;
        frozen_seq_starts = [frozen_seq_starts trial_seq_starts]; %#ok<AGROW>
    end
end

frameTime = cell2mat(frameTime);
ori = cell2mat(ori);
cpd = cell2mat(cpd);

if ip.Results.force_hartley
    zx = find(ori==0);
    iix = randsample(zx, ceil(.5*numel(zx)));
    ori(iix) = 180;
    
    [ky,kx] = pol2cart(ori/180*pi, cpd);
    ori = round(kx, 1);
    cpd = round(ky, 1);
    % randomly assign half the kx==0 conditions to have flipped sign
    % because the space is symmetric
    zx = find(ori==0);
    iix = randsample(zx, ceil(.5*numel(zx)));
    cpd(iix) = -cpd(iix);
end
    
NT = numel(frameTime);

% --- concatenate different sessions
% ts = frameTime(trialStartFrameNum);
% te = frameTime(trialEndFrameNum);

ix = ~isnan(cpd);
cpds = unique(cpd(ix));
oris = unique(ori(ix));
ncpd = numel(cpds);
nori = numel(oris);

[~, oriid] = max(ori==oris', [], 2);
[~, cpdid] = max(cpd==cpds', [], 2);

if ip.Results.force_hartley
    blank = false(size(ori));
%     blank = ori==0 & cpd==0;
else
    blank = cpdid==1;
    ncpd = ncpd - 1;
    cpdid = cpdid - 1;
    cpds(1) = [];
end

% find discontinuous fragments in the stimulus (trials)
% fn_stim   = ceil((te-ts)*opts.fs_stim);

% convert times to samples
% frame_binned = io.convertTimeToSamples(frameTime, opts.fs_stim, ts, fn_stim);

% find frozen repeat indices
opts.frozen_seq_starts = frozen_seq_starts(:);
opts.frozen_seq_dur = frozen_seq_dur;
if ~isempty(opts.frozen_seq_starts)
    opts.frozen_repeats = (bsxfun(@plus, frozen_seq_starts(:), 0:frozen_seq_dur));

    % --- cleanup frozen trials (remove any that are wrong)
    % [sequence, ~, ic] = unique(kx(opts.frozen_repeats), 'rows', 'sorted');
    % opts.frozen_repeats(ic~=1,:) = []; 
    C = nancov([ori(opts.frozen_repeats) cpd(opts.frozen_repeats)]');
    margV = diag(C); C = C ./ sqrt((margV(:)*margV(:)'));
    [~, ref] = max(sum(C > .99, 2));
    keep = C(ref,:) > .99;
    opts.frozen_repeats = opts.frozen_repeats(keep,:);
else
    opts.frozen_repeats = [];
end

opts.oris = oris;
opts.cpds = cpds;
opts.dim =[ncpd nori];

ind = sub2ind(opts.dim, cpdid(~blank), oriid(~blank));

binsize = 1./Exp.S.frameRate;

ft = (1:NT)';
stim{1} = full(sparse(ft(~blank), ind, ones(numel(ind),1), NT, nori*ncpd));
spks = binNeuronSpikeTimesFast(Exp.osp, frameTime, binsize);


% saccade times
sacoffset = ip.Results.num_lags_sac_pre/opts.fs_stim;
stim{2} = full(binTimesFast(Exp.slist(:,1)-sacoffset, frameTime, binsize));
stim{3} = full(binTimesFast(Exp.slist(:,2), frameTime, binsize));


% store eye position at frame


% convert to pixels
if isempty(ip.Results.eye_pos)
    eyeDat(:,2:3) = eyeDat(:,2:3)*Exp.S.pixPerDeg;
else
    eyeDat(:,2:3) = ip.Results.eye_pos*Exp.S.pixPerDeg;
end


% find index into frames
[~, ~,id] = histcounts(frameTime, eyeDat(:,1));
eyeAtFrame = eyeDat(id,2:3);
eyeLabels = Exp.vpx.Labels(id);


t_downsample = ceil(Exp.S.frameRate / opts.fs_stim);
if t_downsample > 1
    stim{1} = downsample_time(stim{1}, t_downsample) / t_downsample;
    stim{2} = downsample_time(stim{2}, t_downsample) / t_downsample;
    stim{3} = downsample_time(stim{3}, t_downsample) / t_downsample;
    frameTime = downsample(frameTime, t_downsample);
    ori = downsample(ori, t_downsample);
    cpd = downsample(cpd, t_downsample);
    eyeAtFrame = downsample_time(eyeAtFrame(valid,:), t_downsample) / t_downsample;
    eyeLabels = downsample_time(eyeLabels(valid), t_downsample) / t_downsample;
    spks = downsample_time(spks, t_downsample);
    frameTime = frameTime(1:size(spks,1));
    ori = ori(1:size(spks,1));
    cpd = cpd(1:size(spks,1));
end

opts.ori = ori;
opts.cpd = cpd;
opts.frameTime = frameTime;
opts.eyeAtFrame = eyeAtFrame;
opts.eyeLabel = eyeLabels;