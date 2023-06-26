function [stat, plotmeta] = fixrate_by_stim(Exp, varargin)
% S = fixrate_by_stim(Exp, varargin)
% Calculate fixation-triggered firing rate for different stimuli
% Inputs:
%   Exp <struct>    experiment struct
% Optional (as argument pairs):
%   'plot'          plot analysis results along the way? (default: true)
%   'win'           time window start and end in seconds (default = [-.1 .5])
%   'binsize'       spike rate bin size (in seconds)
%   'smoothing'     smoothing window (in bins, after binning)
%   'alignto'       align spike rate window to saccade onset or ofset
%                   ('fixon' or 'sacon')
%   'usestim'       use pre or postsaccadic stimulus ('pre' or 'post')
% Output:
%   Stats <struct>
%   PlotMeta <struct>
%
% (c) jly 2020

ip = inputParser();
ip.addParameter('plot', true)
ip.addParameter('win', [-.1 .5])
ip.addParameter('binsize', 1e-3)
ip.addParameter('fixdur', .2)
ip.addParameter('smoothing', 20)
ip.addParameter('alignto', 'fixon', @(x) ismember(x, {'fixon', 'sacon'}))
ip.addParameter('debug', false)
ip.addParameter('stimSets', {'Grating', 'BackImage'})
ip.parse(varargin{:})

alignto = ip.Results.alignto;
binsize = ip.Results.binsize;
win = ip.Results.win;
stimSets = ip.Results.stimSets;
sm = floor(ip.Results.smoothing/2);

lags = win(1):binsize:win(2);
numlags = numel(lags);

cids = Exp.osp.cids;
NC = numel(cids);

% fixation times
fixon = Exp.vpx2ephys(Exp.slist(1:end-1,2));
sacon = Exp.vpx2ephys(Exp.slist(2:end,1));

bad = (fixon+win(1)) < min(Exp.osp.st) | (fixon+win(2)) > max(Exp.osp.st);
fixon(bad) = [];
sacon(bad) = [];

numFixations = numel(fixon);

disp('Binning Spikes')
st = Exp.osp.st;
clu = Exp.osp.clu;
keep = ismember(clu, cids);
st = st(keep);
clu = double(clu(keep));

bs = (st==0) + ceil(st/binsize);
spbn = sparse(bs, clu, ones(numel(bs), 1));
spbn = spbn(:,cids);
blags = ceil(lags/binsize);
switch alignto
    case 'fixon'
        balign = ceil(fixon/binsize);
    case 'sacon'
        balign = ceil(sacon/binsize);
end

% Do the binning here
spks = zeros(numFixations, NC, numlags);
for i = 1:numlags
    spks(:,:,i) = spbn(balign + blags(i),:);
end

disp('Done')

% Get trial number for each fixation
tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(:)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(:)));

[~, trialNum] = getTimeIdx(fixon, tstart, tstop);


stat = struct(); % main output

if ip.Results.plot
   fig = figure(100); clf
end

nm = filtfilt(ones(sm,1)/sm, 1, ones(numlags,1))';

if nargout > 1
    plotmeta = struct();
end

fixdur = sacon - fixon;
for iStim = 1:numel(stimSets)
    
    stimSet = stimSets{iStim};
    validTrials = io.getValidTrials(Exp, stimSet);
    
    ix = ismember(trialNum, validTrials);
    ix = ix & fixdur > ip.Results.fixdur;
    
    stat.(stimSet).numFix = sum(ix);
    
    sp = spks(ix,:,:);
    
    mu = squeeze(mean(sp));
    sd = squeeze(std(sp));
    
    stat.(stimSet).peakloc = nan(NC,1);
    stat.(stimSet).peak = nan(NC,1);
    stat.(stimSet).troughloc = nan(NC,1);
    stat.(stimSet).trough = nan(NC,1);
    stat.(stimSet).lags = lags;
    
    if ip.Results.plot
        sx = ceil(sqrt(NC));
        sy = round(sqrt(NC));
        figure(fig);
    end
    
    if nargout > 1
        plotmeta.(stimSet).lags = lags;
        plotmeta.(stimSet).spks = sp;
        plotmeta.(stimSet).fixdur = fixdur(ix);
    end
    
    
    for cc = 1:NC
    
        spcc = squeeze(sp(:,cc,:));
        spcc = filtfilt(ones(sm,1)/sm, 1, spcc')';
        spcc = spcc ./ nm; % normalize by uneven number of samples
            
        mu(cc,:) = mean(spcc)/binsize;
        sd(cc,:) = std(spcc)/binsize;
        
        m = mu(cc,:);
        m = m ./ median(m) - 1;
        
        [stat.(stimSet).peakloc(cc), stat.(stimSet).peak(cc)] = get_peak_and_trough(lags, m, [-0.05 0.2]);
        [~, ~, stat.(stimSet).troughloc(cc), stat.(stimSet).trough(cc)] = get_peak_and_trough(lags, m, [-0.05 0.1]);
        
        if ip.Results.plot
            subplot(sx,sy,cc)
            h = plot(lags, m); hold on
            plot(stat.(stimSet).peakloc(cc), stat.(stimSet).peak(cc), 'o', 'Color', h.Color, 'MarkerFaceColor', h.Color);
            plot(stat.(stimSet).troughloc(cc), stat.(stimSet).trough(cc), 'v', 'Color', h.Color, 'MarkerFaceColor', h.Color);
        end
        
    end
    stat.(stimSet).meanRate = mu;
    stat.(stimSet).sdRate = sd;
    stat.(stimSet).lags = lags;
end




if ip.Results.plot
    figure(102); clf
    fields = fieldnames(stat);
    cmap = lines;
    for iStim = 1:numel(fields)
        field = fields{iStim};
        plot(stat.(field).peakloc(:), stat.(field).peak(:), 'o', 'Color', cmap(iStim,:), 'MarkerFaceColor', cmap(iStim,:)); hold on
        plot(stat.(field).troughloc(:), stat.(field).trough(:), 'v', 'Color', cmap(iStim,:), 'MarkerFaceColor', cmap(iStim,:));
    end
end
    

