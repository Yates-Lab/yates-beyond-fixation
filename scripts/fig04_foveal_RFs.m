%% add paths
user = 'jakelaptop';
addFreeViewingPaths(user);
addpath Analysis/manuscript_freeviewingmethods/
%% ROIs for fovea sessions
% We have to export the stimulus for offline calibration which uses pytorch
% This code will reconstruct the stimulus at full resolution for relevant
% stimulus paradigms and copy the resulting hdf5 file to the server for
% analysis

% everything reconstructed within an ROI: This is set
RoiS = struct();
RoiS.('logan_20200304') = [-20 -60 50 10];
RoiS.('logan_20200306') = [-20 -60 50 10];
RoiS.('logan_20191231') = [-20 -60 50 10];
RoiS.('logan_20191119') = [-20 -60 50 10];
RoiS.('logan_20191121') = [-20 -50 50 20];

%% load data
sesslist = io.dataFactory();

%%
close all
sessId = sesslist{31}; %'logan_20200304';
spike_sorting = 'kilowf';
[Exp, S] = io.dataFactory(sessId, 'spike_sorting', spike_sorting, 'cleanup_spikes', 0);

eyePosOrig = Exp.vpx.smo(:,2:3);

% correct eye-position offline using the eye position measurements on the
% calibration trials if they were saved
try
    eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);
    
    lam = .5; % mixing between original eye pos and corrected eye pos
    Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);
end


%% get visually driven units
[spkS, W] = io.get_visual_units(Exp, 'plotit', true);

%% plot spatial RFs to try to select a ROI
unit_mask = 0;
NC = numel(spkS);
hasrf = find(~isnan(arrayfun(@(x) x.x0, spkS)));
figure(2); clf
set(gcf, 'Color', 'w')
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

xax = spkS(1).xax/Exp.S.pixPerDeg;
yax = spkS(1).yax/Exp.S.pixPerDeg;
cc = lines(NC);
for cc = 1:NC %hasrf(:)'
    subplot(sx,sy,cc,'align')
    
    
    I = spkS(cc).srf;
    I = I - mean(I);
    I = I / max(abs(I(:)));
    rf = I; %abs(spkS(cc).unit_mask)/max(abs(spkS(cc).unit_mask(:)));
    if isnan(sum(I(:)))
        I = zeros(size(I));
    else
        unit_mask = unit_mask + rf;
    end
    imagesc(xax, yax, I, [-1 1]); hold on
    colormap parula
    axis xy
    plot([0 0], ylim, 'w')
    plot(xlim,[0 0], 'w')
%     xlim([0 40])
%     ylim([-40 0])
%     [~, h] = contour(xax, yax, rf, [.75:.05:1], 'Color', cmap(cc,:)); hold on
end

figure(1); clf
ppd = Exp.S.pixPerDeg;
imagesc(xax*ppd, yax*ppd,unit_mask); axis xy

%%
if isfield(RoiS, sessId)
    S.rect = RoiS.(sessId);
end
figure(1); clf
ppd = Exp.S.pixPerDeg;
imagesc(xax*ppd, yax*ppd,unit_mask); axis xy
hold on
plot(S.rect([1 3]), S.rect([2 2]), 'r', 'Linewidth', 2)
plot(S.rect([1 3]), S.rect([4 4]), 'r', 'Linewidth', 2)
plot(S.rect([1 1]), S.rect([2 4]), 'r', 'Linewidth', 2)
plot(S.rect([3 3]), S.rect([2 4]), 'r', 'Linewidth', 2)
title('Average Spatial RF & ROI')
xlabel('Azimuth (pixels)')
ylabel('Elevation (pixels)')


%% plot some fixation trials

validTrials = io.getValidTrials(Exp, 'FixRsvpStim');
if numel(validTrials) > 1
    n = cellfun(@(x) numel(x.PR.NoiseHistory(:,1)), Exp.D(validTrials)); % trial length
    bad = n < 100;
    validTrials(bad) = [];
    
    % fixation starts / ends
    tstart = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(1), Exp.D(validTrials)));
    tend = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(end,1), Exp.D(validTrials)));
    n(bad) = [];
else
    fprintf("No FixRsvpStim Trials in this dataset\n")
end

%% plot individual trials
if numel(validTrials) > 1
    eyetime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
    iTrial = 0;
    figure(2); clf
    iTrial = iTrial + 1;
    if iTrial > numel(n)
        iTrial = 1;
    end
    
    % iTrial = 9;
    [~, ~, idstart] = histcounts(tstart(iTrial), eyetime);
    [~, ~, idend] = histcounts(tend(iTrial), eyetime);
    
    inds = (idstart - 100):(idend + 100);
    tt = eyetime(inds) - tstart(iTrial);
    
    eyesmoothing = 19;
    eyeX = Exp.vpx.smo(inds,2);
    eyeX = sgolayfilt(eyeX, 1, eyesmoothing);
    eyeY = Exp.vpx.smo(inds,3);
    eyeY = sgolayfilt(eyeY, 1, eyesmoothing);
    plot(tt, eyeX*60, 'k'); hold on
    plot(tt, eyeY*60, 'Color', .5*[1 1 1])
    axis tight
    ylim([-1 1]*60)
    xlabel('Time (s)')
    ylabel('Arcmin')
    plot.fixfigure(gcf, 8, [4 2])
    title(iTrial)
    plot(xlim, [1 1]*30, 'r--')
    plot(xlim, -[1 1]*30, 'r--')
else
    fprintf("No FixRsvpStim Trials in this dataset\n")
end

%% regenerate data with the following parameters
close all

% pixels run down so enforce this here
S.rect([2 4]) = sort(-S.rect([2 4]));

fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Dots'}, 'overwrite', true);

%% copy to server
server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'; %/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
command = 'scp ';
command = [command fname ' '];
command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname)

%% test that it worked
id = 1;
stim = 'Dots';
tset = 'Train';

sz = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'size');

iFrame = 1;

%% show sample frame
iFrame = iFrame + 1;
I = h5read(fname, ['/' stim '/' tset '/Stim'], [iFrame, 1,1], [1 sz(1:2)']);
I = squeeze(I);
% I = h5read(fname{1}, ['/' stim '/' set '/Stim'], [1,1,iFrame], [sz(1:2)' 1]);
figure(id); clf
subplot(1,2,1)
imagesc(I)
subplot(1,2,2)
imagesc(I(1:2:end,1:2:end));% axis xy
colorbar
colormap gray


%% get STAs to check that you have the right rect

Stim = h5read(fname, ['/' stim '/' tset '/Stim']);
% Robs = h5read(fname, ['/' stim '/' set '/Robs']);
ftoe = h5read(fname, ['/' stim '/' tset '/frameTimesOe']);

frate = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'frate');
st = h5read(fname, ['/Neurons/' spike_sorting '/times']);
clu = h5read(fname, ['/Neurons/' spike_sorting '/cluster']);
cids = h5read(fname, ['/Neurons/' spike_sorting '/cids']);
sp.st = st;
sp.clu = clu;
sp.cids = cids;
Robs = binNeuronSpikeTimesFast(sp, ftoe-8e-3, 1/frate);

% Robs = 
eyeAtFrame = h5read(fname, ['/' stim '/' tset '/eyeAtFrame']);
labels = h5read(fname, ['/' stim '/' tset '/labels']);
NX = size(Stim,2);
NY = size(Stim,3);
%%

Stim = reshape(Stim, size(Stim, 1), NX*NY);

% reshape(Stim, 
Stim = zscore(single(Stim));

%% forward correlation
NC = size(Robs,2);
nlags = 20;
Rdelta = Robs - mean(Robs);
nstim = size(Stim,2);
ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc < 5.2 & labels == 1;

stas = zeros(nlags, nstim, NC);
for idim = 1:nstim
    fprintf('%d/%d\n', idim, nstim)
    Xstim = conv2(Stim(:,idim), eye(nlags), 'full');
    Xstim = Xstim(1:end-nlags+1,:);
    stas(:, idim, :) = Xstim(ix,:)'*Rdelta(ix,:);
end

% %%
% % only take central eye positions
% ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
% ix = ecc > 5.1 | labels ~= 1;
% 
% NC = size(Robs,2);
% nlags = 10;
% NT = size(Stim,1);
% sx = ceil(sqrt(NC));
% sy = round(sqrt(NC));
% 
% figure(1); clf
% stas = zeros(nlags, size(Stim,2), NC);
% 
% for cc = 1:NC
%     if sum(Robs(:,cc))==0
%         continue
%     end
%     rtmp = Robs(:,cc);
%     rtmp(ix) = 0; % remove spikes we don't want to analyze
%     rtmp = rtmp - mean(rtmp);
%     sta = simpleRevcorr(Stim, rtmp, nlags);
%     subplot(sx, sy, cc, 'align')
%     plot(sta)
%     stas(:,:,cc) = sta;
%     drawnow
% end
% 
% %%
% % clearvars -except Exp Stim Robs
%%
cc = 0;

%% plot one by one
figure(2); clf
cc = cc + 1;
if cc > NC
    cc = 1;
end

sta = stas(:,:,cc);
% NY = size(Stim,2)/NX;
% sta = (sta - min(sta(:))) ./ (max(sta(:)) - min(sta(:)));
sta = (sta - mean(sta(:))) ./ std(sta(:));
% x = xax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
% y = yax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
% xax = (1:NX)/Exp.S.pixPerDeg*60;
% yax = (1:NY)/Exp.S.pixPerDeg*60;
for ilag = 1:nlags
   subplot(1,nlags, ilag, 'align')
   imagesc(reshape(sta(ilag,:), [NY NX])', [-1 1]*4)
end

% colormap(plot.viridis)
colormap(gray)
title(cc)


%%
figure(2); clf
cc = cc + 1;
NC = numel(W);
if cc > NC
    cc = 1;
end
sta = stas(:,:,W(cc).cid);
sta = (sta - mean(sta(:))) ./ std(sta(:));
[~, bestlag] = max(std(sta,[],2));
figure(1); clf

subplot(2,2,1)
plot(W(cc).wavelags, W(cc).ctrChWaveform, 'k'); hold on
plot(W(cc).wavelags, W(cc).ctrChWaveformCiHi, 'k--')
plot(W(cc).wavelags, W(cc).ctrChWaveformCiLo, 'k--')
title(cc)
axis tight
ax = subplot(2,2,3);
plot(W(cc).lags, W(cc).isi)
title(W(cc).isiV)
% ax.XScale ='log';

subplot(2,2,2)
imagesc(reshape(sta(bestlag,:), [NX NY])')

[~, imx] = max(sta(bestlag,:));
[~, imn] = min(sta(bestlag,:));

subplot(2,2,4)
plot(sta(:,imx), 'b'); hold on
plot(sta(:,imn), 'r')


%% explore waveforms
cids = [0,2,3,8,9,10,11,15,18,19,21,22,23,24,26,27,29,30,31,32,34,35,36,37,38,40,41,42,43,45,48,52,56,57,58,60,63,64]+1;
figure(10); clf
plot.plotWaveforms(W, 1, 'cids', cids)

%%
sid = regexp(sessId, '_', 'split');

fname2 = ['Figures/2021_pytorchmodeling/rfs_' sid{2} '_' spike_sorting '.mat'];
if exist(fname2, 'file')
    tmp = load(fname2);
end

% tmp.shiftx = flipud(tmp.shiftx);
% tmp.shifty = flipud(tmp.shifty);

tmp.shiftx = tmp.shiftx / 60 * 5.2;
tmp.shifty = tmp.shifty / 60 * 5.2;
%%

figure(2); clf
subplot(1,2,1)
contourf(tmp.xspace, tmp.yspace, tmp.shiftx, 'w'); colorbar
subplot(1,2,2)
contourf(tmp.xspace, tmp.yspace, tmp.shifty, 'w'); colorbar

%%
% 
eyeAtFrame = h5read(fname, ['/' stim '/' tset '/eyeAtFrame']);
eyeX = (eyeAtFrame(:,2) - Exp.S.centerPix(1)) / Exp.S.pixPerDeg;
eyeY = (eyeAtFrame(:,3) - Exp.S.centerPix(2)) / Exp.S.pixPerDeg;

figure(1); clf
subplot(2,1,1)
plot(eyeX); hold on
plot(eyeY)

shiftX = interp2(tmp.xspace, tmp.yspace, tmp.shiftx, eyeX, eyeY, 'linear', 0);
shiftY = interp2(tmp.xspace, tmp.yspace, tmp.shifty, eyeX, eyeY, 'linear', 0);

subplot(2,1,2)
plot(shiftX); hold on
plot(shiftY)

%%
Stim = h5read(fname, ['/' stim '/' tset '/Stim']);
StimS = zeros(size(Stim), 'like', Stim);
%%
dims = size(Stim);
NT = dims(1);
dims(1) = [];

% xax = floor(-dims(2)/2:dims(2)/2)/Exp.S.pixPerDeg*60;
% yax = floor(-dims(1)/2:dims(1)/2)/Exp.S.pixPerDeg*60;
xax = linspace(-1,1,dims(2));
yax = linspace(-1,1,dims(1));
[xgrid,ygrid] = meshgrid(xax(1:dims(2)), yax(1:dims(1)));

disp("shift correcting stimulus...")
for iFrame = 1:NT
    xsample = xgrid+shiftX(iFrame);
    ysample = ygrid+shiftY(iFrame);
    I = interp2(xgrid, ygrid, single(squeeze(Stim(iFrame,:,:))), xsample, ysample, 'linear', 0);
    StimS(iFrame,:,:) = I;
end
disp("Done")

% I0 = single(squeeze(Stim(iFrame,:,:)));
% figure(1); clf
% subplot(1,3,1)
% imagesc(I0)
% subplot(1,3,2)
% imagesc(I)
% subplot(1,3,3)
% imagesc(I0-I)

%%
NC = size(Robs,2);
nlags = 20;
Rdelta = Robs - mean(Robs);
nstim = size(Stim,2);
ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc < 5.2 & labels == 1;
ix = sum(conv2(double(ix), eye(nlags), 'full'),2) == nlags; % all timelags good

stas = zeros(nlags, dims(1), dims(2), NC);
for ii = 1:dims(1)
    fprintf('%d/%d\n', ii, dims(1))
    for jj = 1:dims(2)
        Xstim = conv2(StimS(:,ii,jj), eye(nlags), 'full');
        Xstim = Xstim(1:end-nlags+1,:);
        stas(:, ii,jj, :) = Xstim(ix,:)'*Rdelta(ix,:);
    end
end

%%
% clf
% plot.plotWaveforms(W)

cmap = [[ones(128,1) repmat(linspace(0,1,128)', 1, 2)]; [flipud(repmat(linspace(0,1,128)', 1, 2)) ones(128,1)]];
xax = linspace(-1,1,NX)*30;
yax = linspace(-1,1,NY)*30;

figure(1); clf
for cc = 1:NC
%     sta = stas(:,:,:,W(cc).cid);
    sta = tmp.stas_post(:,:,:,cc);
    nlags = size(sta,1);
    sta = reshape(sta, nlags, []);
    sta = (sta - mean(sta(:))) ./ std(sta(:));
    [bestlag, ~] = find(max(abs(sta(:)))==abs(sta));
    extrema = max(max(sta(:)), max(abs(sta(:))));
    I = reshape(sta(bestlag,:), [NX NY])';
    ind = 10:60;
    I = I(ind, ind);
%     abs(I).^10
    imagesc(xax(ind) + W(cc).x, yax(ind) + W(cc).depth, I, .5*[-1 1]*extrema); hold on
    xlim([-50 250])
    ylim([-50 1.1*max([W.depth])])
    drawnow
    
end

colormap(cmap)

%%
figure(2); clf
cc = cc + 1;
NC = numel(W);
if cc > NC
    cc = 1;
end

sta = stas(:,:,:,W(cc).cid);


% sta = tmp.stas_post(:,:,:,cc);
% nlags = size(tmp.stas_pre,1);

nlags = size(sta,1);
sta = reshape(sta, nlags, []);
sta = (sta - mean(sta(:))) ./ std(sta(:));
[bestlag, ~] = find(max(abs(sta(:)))==abs(sta));
% [~, bestlag] = max(max(sta,[],2));
figure(1); clf

subplot(2,2,1)
plot(W(cc).wavelags, W(cc).ctrChWaveform, 'k'); hold on
plot(W(cc).wavelags, W(cc).ctrChWaveformCiHi, 'k--')
plot(W(cc).wavelags, W(cc).ctrChWaveformCiLo, 'k--')
title(cc)
axis tight
ax = subplot(2,2,3);
plot(W(cc).lags, W(cc).isi)
title(W(cc).isiV)
% ax.XScale ='log';

extrema = max(max(sta(:)), max(abs(sta(:))));
subplot(2,2,2)
imagesc(reshape(sta(bestlag,:), [NX NY])', [-1 1]*extrema)
% colormap(plot.viridis)
colormap parula
% axis xy

[~, imx] = max(sta(bestlag,:));
[~, imn] = min(sta(bestlag,:));

subplot(2,2,4)
plot(sta(:,imx), 'b'); hold on
plot(sta(:,imn), 'r')


figure(3); clf
for ilag = 1:nlags
    subplot(1,nlags,ilag)
    imagesc(reshape(sta(ilag,:), [NX NY])', [-1 1]*extrema)
end