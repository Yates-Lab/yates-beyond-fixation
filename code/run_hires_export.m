function run_hires_export(Exp, varargin)
% run_hires_export
% Takes in an Exp struct and saves a 

ip = inputParser();
ip.addParameter('fig_dir', '.')
ip.addParameter('fig_name', [])
ip.parse(varargin{:})

%% Step 1: post-process
Exp = fix_mitchelllab_exports(Exp);


%% set up figure saving
if isempty(ip.Results.fig_name)
    figname = Exp.FileTag;
    
else
    figname = ip.Results.fig_name;
end

[~, figname, ~] = fileparts(figname);
processedFileName = figname;

figname = [figname '.pdf'];

figname = fullfile(ip.Results.fig_dir, sprintf('hiresexport_%s.pdf', figname));

%% plot gaze position by condition
binsize = .1;
sigma = .25;
bins = -15:binsize:15;

PRname = cellfun(@(x) x.PR.name, Exp.D, 'uni', 0);
PRnames = unique(PRname);

eyeTime = Exp.vpx2ephys(Exp.vpx.raw(:,1));

figure(1); clf
set(gcf, 'Color', 'w')
N = numel(PRnames);
sx = ceil(sqrt(N));
sy = round(sqrt(N));
for i = 1:N
    subplot(sx, sy, i)

    idx = find(strcmp(PRname, PRnames{i}));
    start = cellfun(@(x) x.START_EPHYS, Exp.D(idx));
    stop = cellfun(@(x) x.END_EPHYS, Exp.D(idx));

    iix = find(getTimeIdx(eyeTime, start, stop));

    C = histcounts2(Exp.vpx.fix(iix,1), Exp.vpx.fix(iix,2), bins, bins)';
    C = imgaussfilt(C, ceil(sigma/binsize));

    imagesc(bins, bins, log10(C))
    colormap(parula)
    axis xy
    title(PRnames{i})
end

set(gcf, 'PaperSize', [8.5 11], 'PaperPosition', [0 0 7 7])
exportgraphics(gcf, figname);

%% get coarse resolution spatial RFs

% these data come from the foveal representation so the central 3 d.v.a
% are sufficien
ROI = [-1 -1 1 1]*3;
% coarse bin size
binSize = .25;
Frate = 120;

eyeposexclusion = 20;
win = [-1 20];
ROIWINDOWSIZE = 150; % spatial dimensions of the high-res ROI

[Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
    'ROI', ROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
    'eyePosExclusion', eyeposexclusion * Exp.S.pixPerDeg, ...
    'eyePos', Exp.vpx.fix, 'frate', Frate, ...
    'fastBinning', true, ...
    'smoothing', 2);

% use indices only while eye position is on the screen
scrnBnds = (Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg;
scrnBnds = 1.5 * scrnBnds;
eyePosAtFrame = opts.eyePosAtFrame/Exp.S.pixPerDeg;

ix = (eyePosAtFrame(:,1) + ROI(1)) >= -scrnBnds(1) & ...
    (eyePosAtFrame(:,1) + ROI(3)) <= scrnBnds(1) & ...
    (eyePosAtFrame(:,2) + ROI(2)) >= -scrnBnds(2) & ...
    (eyePosAtFrame(:,2) + ROI(4)) <= scrnBnds(2);

fprintf('%02.2f%% of gaze positions are safely on screen\n', 100*mean(ix))

numspikes = sum(RobsSpace(ix,:));

% forward correlation
stas = forwardCorrelation(full(Xstim), sum(RobsSpace-mean(RobsSpace, 1),2), win, find(ix), [], true, false);

close all
figure(1); clf
set(gcf, 'Color', 'w')
% stas = forwardCorrelation(Xstim, mean(RobsSpace,2), win);
stas = stas / std(stas(:)) - mean(stas(:));
wm = [min(stas(:)) max(stas(:))];
nlags = size(stas,1);
for ilag = 1:nlags
    subplot(2, ceil(nlags/2), ilag)
    imagesc(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg, reshape(stas(ilag, :), opts.dims), wm)
    title(sprintf('lag: %02.2f', ilag*(1000/Frate)))
    axis xy
end

set(gcf, 'PaperSize', [8.5 11], 'PaperPosition', [0 0 8 3])
exportgraphics(gcf, figname, 'Append', true);

%% find ROI

winsize = ROIWINDOWSIZE;
rf = reshape(std(stas), opts.dims);
rf = (rf - min(rf(:))) / (max(rf(:)) - min(rf(:)));
[con, ar, ctr] = get_rf_contour(opts.xax, opts.yax, rf, 'thresh', .7);
imagesc(opts.xax, opts.yax, rf);
hold on
plot(ctr(1), ctr(2), 'or')
S.rect = round([ctr ctr]) + [-1 -1 1 1]*winsize/2;


in_pixels = false;
if in_pixels
    ppd = 1;
else
    ppd = Exp.S.pixPerDeg;
end

figure(2); clf
imagesc(opts.xax/ppd, opts.yax/ppd, rf)
axis xy

hold on
plot(S.rect([1 3])/ppd, S.rect([2 2])/ppd, 'r', 'Linewidth', 2)
plot(S.rect([1 3])/ppd, S.rect([4 4])/ppd, 'r', 'Linewidth', 2)
plot(S.rect([1 1])/ppd, S.rect([2 4])/ppd, 'r', 'Linewidth', 2)
plot(S.rect([3 3])/ppd, S.rect([2 4])/ppd, 'r', 'Linewidth', 2)
title('Average Spatial RF & ROI')
xlabel('Azimuth (pixels)')
ylabel('Elevation (pixels)')

plot.fixfigure(gcf, 12, [4 4]);
% saveas(gcf, fullfile('figures/hires_export', sprintf('%s_roi.pdf', strrep(processedFileName, '.mat', ''))) )
% pixels run down so enforce this here
S.rect([2 4]) = sort(-S.rect([2 4]));

set(gcf, 'PaperSize', [8.5 11], 'PaperPosition', [0 0 4 3])
exportgraphics(gcf, figname, 'Append', true);


%% Do high-res reconstruction using PTB (has to replay the whole experiment)
Exp.FileTag = processedFileName;
S.spikeSorting = 'kilo';
stimsets = {'Dots', 'Gabor', 'BackImage', 'Grating', 'FixRsvpStim', 'FixFlashGabor'};
%     fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Gabor'}, 'overwrite', false, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false, 'EyeSmoothing', 19, 'EyeSmoothingOrder', 1);
fname = make_stimulus_file_for_py(Exp, S, 'stimlist', stimsets, 'overwrite', false, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false, 'useFix', true);

%% get STAs to check that you have the right rect
stim = 'Gabor';
tset = 'Train';
spike_sorting = 'kilo';
Stim = h5read(fname, ['/' stim '/' tset '/Stim']);
% Robs = h5read(fname, ['/' stim '/' set '/Robs']);
ftoe = h5read(fname, ['/' stim '/' tset '/frameTimesOe']);

frate = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'frate');
st = h5read(fname, ['/Neurons/' spike_sorting '/times']);
clu = h5read(fname, ['/Neurons/' spike_sorting '/cluster']);
cids = h5read(fname, ['/Neurons/' spike_sorting '/cids']);
sp = struct();
sp.st = st;
sp.clu = clu;
sp.cids = cids;
Robs = binNeuronSpikeTimesFast(sp, ftoe-8e-3, 1/frate);

% Robs =
eyeAtFrame = h5read(fname, ['/' stim '/' tset '/eyeAtFrame']);
labels = h5read(fname, ['/' stim '/' tset '/labels']);
NX = size(Stim,2);
NY = size(Stim,3);
NC = size(Robs,2);

Stim = reshape(Stim, size(Stim, 1), NX*NY);
Stim = zscore(single(Stim));



%% Pick a lag and compute the STA quickly for all cells

lag = 8;
ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc < 5.2 & labels == 1 & (1:numel(ecc))'> lag;
Rdelta = Robs - mean(Robs);
Rdelta = Rdelta(ix,:);
sta = (Stim(find(ix)-lag,:).^2)'*Rdelta;
% sta = (Stim(find(ix)-lag,:))'*Rdelta;
[~, ind] = sort(std(sta));

sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(10); clf
for cc = 1:NC
    subplot(sx, sy, cc, 'align')
    imagesc(reshape(sta(:,ind(cc)), [NX NY]))
    axis off
end

set(gcf, 'PaperSize', [7 7], 'PaperPosition', [0 0 7 7])
exportgraphics(gcf, figname, 'Append', true);

figure(11); clf
plot(std(sta(:,ind)), '-o'); hold on
thresh = median(std(sta));
cids = find(std(sta) > thresh);
plot(xlim, thresh*[1 1], 'r')
set(gcf, 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
exportgraphics(gcf, figname, 'Append', true);

%% try to find center of gaze

% we need the trial starts and stops and protocol name
% we need raw eye traces
% we need meta data that we're interest in (e.g., Grating / Probe / Reward)


%% write eye position to h5 file

% figure(1); clf
% plot(Exp.vpx2ephys(Exp.vpx.smo(:,1)), Exp.vpx.smo(:,2)); hold on
% plot(ftoe, (eyeAtFrame(:,2)-Exp.S.centerPix(1))/Exp.S.pixPerDeg, '.')

Stat = struct();
Stat.timestamps = Exp.vpx2ephys(Exp.vpx.smo(:,1));
Stat.eyeposDeg = Exp.vpx.smo(:,2:3);
Stat.ppd = Exp.S.pixPerDeg;
Stat.ctrpx = Exp.S.centerPix;

fname = io.h5_add_struct(fname, Stat, '/ddpi');

%% write trial starts, stops, protocol, and reward

Stat = struct();

Stat.trialStarts = cellfun(@(x) x.START_EPHYS, Exp.D);
Stat.trialStops = cellfun(@(x) x.END_EPHYS, Exp.D);
nTrials = numel(Exp.D);
Stat.protocol = repmat({'unrecognized'}, nTrials, 1);
Stat.rewardTimes = Exp.ptb2Ephys(cell2mat(cellfun(@(x) x.rewardtimes', Exp.D, 'UniformOutput', false)));

stimList = {'Grating', 'Gabor', 'Dots', 'BackImage', ...
    'Forage', ...
    'FixRsvpStim', ...
    'FaceCal', ...
    'FixCalib', ...
    'ForageStaticLines', ...
    'FixFlashGabor', ...
    'MTDotMapping', ...
    'DriftingGrating'};

for i = 1:numel(stimList)
    tlist = io.getValidTrials(Exp, stimList{i});
    if isempty(tlist)
        continue
    end
    for j = tlist(:)'
        Stat.protocol{j} = stimList{i};
    end
end

fname = io.h5_add_struct(fname, Stat, '/trials');
%%


% %%
% [m, s, bc, v, tspcnt] = eventPsth(Exp.vpx2ephys(Exp.slist(:,1)), Stat.rewardTimes, [-.5 .5], .005);
%
% figure(1); clf
% plot(bc, m)

%% write the imagefiles and


tlist = io.getValidTrials(Exp, 'BackImage');
imstarts = Exp.ptb2Ephys(cellfun(@(x) x.PR.startTime, Exp.D(tlist)));
imstops = Exp.ptb2Ephys(cellfun(@(x) x.PR.imageOff, Exp.D(tlist)));
imrect = cell2mat(cellfun(@(x) x.PR.destRect, Exp.D(tlist), 'uni', 0));
imlist = [];
for i = tlist(:)'
    tmp = strsplit(Exp.D{i}.PR.imagefile, '/');
    imlist = [imlist; tmp(end)];
end

Stat = struct();
Stat.startTimes = imstarts;
Stat.stopTimes = imstops;
Stat.imageList = imlist;
Stat.imageRect = imrect;
fname = io.h5_add_struct(fname, Stat, '/BackImage');

% f = h5info(fname, '/BackImage');
% arrayfun(@(x) x.Name, f.Groups(2).Datasets, 'uni', 0)


%% write the Grating condition

tlist = io.getValidTrials(Exp, 'Grating');

% isfrozen = cellfun(@(x) x.PR.frozenSequence==1, Exp.D(tlist));

Stat = struct();
Stat.timestamps = Exp.ptb2Ephys(cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,1), Exp.D(tlist), 'uni', 0)));
Stat.ori = cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,2), Exp.D(tlist), 'uni', 0));
Stat.sf = cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,3), Exp.D(tlist), 'uni', 0));

% figure(1); clf
% plot(Stat.timestamps, Stat.ori, '.')

fname = io.h5_add_struct(fname, Stat, '/Grating');


%% plot repeats condition so we can evaluate if a session is good

idx = io.getValidTrials(Exp, 'FixRsvpStim');
if isempty(idx)
    return
end


num_trials = numel(idx);
stim_start = nan(num_trials, 1);
trial_start = nan(num_trials,1);
stim_end = nan(num_trials,1);
for i = 1:num_trials
    if ~isempty(Exp.D{idx(i)}.PR.NoiseHistory)
        stim_start(i) = Exp.D{idx(i)}.PR.NoiseHistory(1,1);
        stim_end(i) = Exp.D{idx(i)}.PR.NoiseHistory(end,1);
        trial_start(i) = Exp.D{idx(i)}.START_EPHYS;
    end
    
end


dur = stim_end - stim_start;
ix = dur > .5;
stim_start = stim_start(ix);
stim_end = stim_end(ix);
dur = dur(ix);

stim_start = Exp.ptb2Ephys(stim_start);
stim_end = Exp.ptb2Ephys(stim_end);

win = [-.2 max(dur)];

bin_size = 1e-3;

cids = unique(Exp.osp.clu);


NC = numel(cids);
nperfig = 20;
nfigs = ceil(NC/nperfig);

sx = ceil(sqrt(nperfig));
sy = round(sqrt(nperfig));

for cc = 1:NC
    fignum = ceil(cc/nperfig);
    figure(fignum)
    
    iplot = mod(cc, nperfig);
    if iplot==0
        iplot=nperfig;
    end
    
    sptimes = Exp.osp.st(Exp.osp.clu==cids(cc));
    [~,~,bc,~,spcnt] = eventPsth(sptimes, stim_start, win, bin_size);
    [~, ind] = sort(dur);
    [i,j] = find(spcnt(ind,:));
    subplot(sx, sy, iplot)
    plot.raster(bc(j),i); hold on
    plot(dur(ind), 1:numel(ind))
    xlim(win)
    title(cc)
end

for i = 1:nfigs
    exportgraphics(figure(i), figname, 'Append', true);
end
