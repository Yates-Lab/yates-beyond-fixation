
%% Load session

datadir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'hires');
outputdir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'stim_movies');
sesslist = arrayfun(@(x) x.name, dir(fullfile(datadir, '*.mat')), 'uni', 0);

flist = dir(fullfile(outputdir, '*.hdf5'));

% Loop over sessions and import (this will be super slow)
for isess = 1:numel(sesslist)
    processedFileName = sesslist{isess};
    fname = fullfile(datadir, processedFileName);
    sessname = strrep(processedFileName, '.mat', '');

    exportexists=sum(arrayfun(@(x) contains(x.name, sessname), flist))>0;
    fprintf('%d) %d\n', isess, exportexists)

    %     assert(sum(arrayfun(@(x) contains(x.name, sessname), flist))==0, 'export_hires: session already exists')

    assert(exist(fname, 'file'), "export_hires: preprocessed file does not exist. run step 01 first")
    if exportexists
        disp('Export exists. skipping')
        continue
    end
    
    Exp = load(fname);

    %% get RFs

    ROI = [-1 -1 1 1]*3;
    binSize = .25;
    Frate = 120;
    eyeposexclusion = 20;
    win = [-1 20];

    [Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
        'ROI', ROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
        'eyePosExclusion', eyeposexclusion * Exp.S.pixPerDeg, ...
        'eyePos', Exp.vpx.smo(:,2:3), 'frate', Frate, ...
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
    % stas = forwardCorrelation(Xstim, mean(RobsSpace,2), win);
    stas = stas / std(stas(:)) - mean(stas(:));
    wm = [min(stas(:)) max(stas(:))];
    nlags = size(stas,1);
    for ilag = 1:nlags
        subplot(2, ceil(nlags/2), ilag)
        imagesc(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg, reshape(stas(ilag, :), opts.dims), wm)
        title(sprintf('lag: %02.2f', ilag*16))
        axis xy
    end

    %% find ROI

    winsize = 80;
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
    saveas(gcf, fullfile('figures/hires_export', sprintf('%s_roi.pdf', strrep(processedFileName, '.mat', ''))) )


    %% Do high-res reconstruction using PTB (has to replay the whole experiment)
    Exp.FileTag = processedFileName;
    % pixels run down so enforce this here
    S.rect([2 4]) = sort(-S.rect([2 4]));
    
    fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Dots', 'Gabor', 'BackImage', 'Grating', 'FixRsvpStim'}, 'overwrite', false, 'GazeContingent', true, 'includeProbe', true);

end

%% Copy to server
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
stim = 'FixFlashGabor';
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
drawnow



%% get STAs to check that you have the right rect
spike_sorting = 'kilo';
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
   subplot(2,ceil(nlags/2), ilag, 'align')
   imagesc(reshape(sta(ilag,:), [NX NY])', [-1 1]*4)
end

% colormap(plot.viridis)
colormap(gray)
title(cc)


%%
figure(2); clf
cc = cc + 1;
% NC = numel(W);
if cc > NC
    cc = 1;
end
% cc = 39;

sta = stas(:,:,cc);
sta = (sta - mean(sta(:))) ./ std(sta(:));
[~, bestlag] = max(std(sta,[],2));


subplot(2,2,2)
imagesc(reshape(sta(bestlag,:), [NX NY]))
title('Standard Fixation')

[~, imx] = max(sta(bestlag,:));
[~, imn] = min(sta(bestlag,:));

subplot(2,2,4)
plot(sta(:,imx), 'b'); hold on
plot(sta(:,imn), 'r')

title(cc)

sta = stasShift(:,:,cc);
sta = (sta - mean(sta(:))) ./ std(sta(:));

subplot(2,2,1)
imagesc(reshape(sta(bestlag,:), [NX NY]))
title('Gaze Corrected')

[~, imx] = max(sta(bestlag,:));
[~, imn] = min(sta(bestlag,:));

subplot(2,2,3)
plot(sta(:,imx), 'b'); hold on
plot(sta(:,imn), 'r')
