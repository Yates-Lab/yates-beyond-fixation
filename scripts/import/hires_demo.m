
%% Load session

datadir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'hires');
outputdir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'stim_movies');
sesslist = arrayfun(@(x) x.name, dir(fullfile(datadir, '*.mat')), 'uni', 0);
flist = dir(fullfile(outputdir, '*.hdf5'));

%% Loop over sessions and import (this will be super slow)
sesslist = {'allen_20220610.mat'};

for isess = 1
    processedFileName = sesslist{isess};
    fname = fullfile(datadir, processedFileName);
    sessname = strrep(processedFileName, '.mat', '');

    exportexists=sum(arrayfun(@(x) contains(x.name, sessname), flist))>0;
    fprintf('%d) [%s] exported = %d\n', isess, sessname, exportexists)

    %     assert(sum(arrayfun(@(x) contains(x.name, sessname), flist))==0, 'export_hires: session already exists')

%     assert(exist(fname, 'file'), "export_hires: preprocessed file does not exist. run step 01 first")
%     if exportexists
%         disp('Export exists. skipping')
%         continue
%     end
    
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
    % pixels run down so enforce this here
    S.rect([2 4]) = sort(-S.rect([2 4]));

    %% Do high-res reconstruction using PTB (has to replay the whole experiment)
    Exp.FileTag = processedFileName;
    S.spikeSorting = 'kilowf';
%     {'Dots', 'Gabor', 'BackImage', 'Grating', 'FixRsvpStim'}
    fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Gabor', 'BackImage'}, 'overwrite', false, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false);

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
stim = 'Gabor';
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
spike_sorting = 'kilowf';
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
%%

Stim = reshape(Stim, size(Stim, 1), NX*NY);

% reshape(Stim, 
Stim = zscore(single(Stim));

%% Pick a lag and compute the STA quickly for all cells 
lag = 8;
ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc < 5.2 & labels == 1;
Rdelta = Robs - mean(Robs);
Rdelta = Rdelta(ix,:);
sta = (Stim(find(ix)-lag,:).^2)'*Rdelta;
[~, ind] = sort(std(sta));

sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(10); clf
for cc = 1:NC
    subplot(sx, sy, cc, 'align')
    imagesc(reshape(sta(:,ind(cc)), [NX NY]))
    axis off
end

figure(11); clf
plot(std(sta(:,ind)), '-o')
cids = find(std(sta) > 500); 

%%
xax = -5:.5:5;

eyeX = (eyeAtFrame(:,2)-Exp.S.centerPix(1))/Exp.S.pixPerDeg;
eyeY = (eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;

R = Robs(:,cids);
Rbar = mean(Rbar);

n = numel(xax);
win = 1.5;

figure(10); clf
ctrs = nan(n, n, 2);
threshs = nan(n, n);

axs = plot.tight_subplot(n, n, .01, .01);

for ii = 1:n
    for jj = 1:n
        
        ix = (hypot(eyeX - xax(ii), eyeY - xax(jj)) < win) & labels == 1;
        fprintf('x: %.2f, y: %0.2f, %d samples\n', xax(ii), xax(jj), sum(ix))
        if sum(ix) < 200
            continue
        end
        
        ix = find(ix);
        ix = ix(ix > lag);
        Rdelta = R(ix,:) - Rbar;
        sta = (Stim(ix-lag,:).^2)'*Rdelta;

%         subplot(n, n, (ii-1)*n + jj)
        set(gcf, 'currentaxes', axs((jj-1)*n + ii))
        I = reshape(mean(sta,2), [NX, NY]);
        I = imgaussfilt(I,2);
        I = (I - min(I(:))) / (max(I(:))- min(I(:)));
        imagesc(I); colormap gray
        axis off
        hold on
        [con, ar, ctr, thresh] = get_rf_contour(1:NX, 1:NY, I);

        if isnan(ctr(1))
            continue
        end
        ctrs(jj,ii,:) = ctr;
        
        plot(con(:,1), con(:,2), 'r', 'Linewidth', 2)
        threshs(jj,ii) = thresh;

        axis off
        drawnow
    end

end

%% plot map of errors
figure(11); clf
subplot(1,2,1)
imagesc(ctrs(:,:,1).*(threshs<.8))
subplot(1,2,2)
imagesc(ctrs(:,:,2).*(threshs<.8))

%%

[xx, yy] = meshgrid(xax);

figure(1); clf;
[i,j] = find(xx==0 & yy==0);
cx = ctrs(i, j, 1);
cy = ctrs(i, j, 2);

imagesc()
mask = (threshs<.8);
errX = ctrs(:,:,1).*mask-cx;
errY = ctrs(:,:,2).*mask-cy;

plot3(xx(:), yy(:), errX(:), '.'); hold on

% fit a plane
fun = @(theta, x) (x - [theta(1) theta(2)])*[theta(3); theta(4)];

theta0 = [0 0 .5 .5];
thHatX = lsqcurvefit(fun, theta0, [xx(mask) yy(mask)], errX(mask));
thHatY = lsqcurvefit(fun, theta0, [xx(mask) yy(mask)], errY(mask));

xHat = fun(thHatX, [xx(:) yy(:)]);
yHat = fun(thHatY, [xx(:) yy(:)]);

plot3(xx(:), yy(:), xHat(:), '.')

for i = find(mask)'
    plot3([xx(i) xx(i)], [yy(i) yy(i)], [errX(i) xHat(i)], 'k-'); hold on
end

figure(2); clf,
subplot(1,2,1)
imagesc(errX, [-20 20])
subplot(1,2,2)
imagesc(reshape(xHat, [n n]), [-20 20])

figure(3); clf,
subplot(1,2,1)
imagesc(errY, [-20 20])
subplot(1,2,2)
imagesc(reshape(yHat, [n n]), [-20 20])

%%
[xx,yy] = meshgrid(1:NX, 1:NY);

NT = size(Stim, 1);
X = reshape(Stim, [NT, NX, NY]);
shiftX = fun(thHatX, [eyeX, eyeY]);
shiftY = fun(thHatY, [eyeX, eyeY]);
V = zeros(size(X));

for i = 1:NT
    V(i,:,:) = interp2(squeeze(X(i,:,:)), shiftX(i)+xx, shiftY(i)+yy, 'nearest', 0);
end
disp("Done")

%%
lag = 8;
ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc < 5.2 & labels == 1;
Rdelta = Robs - mean(Robs);
Rdelta = Rdelta(ix,:);
X = reshape(V, size(Stim));
X(isnan(X))=0;
sta = (X(find(ix)-lag,:))'*Rdelta;
[~, ind] = sort(std(sta));

sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(12); clf
for cc = 1:NC
    subplot(sx, sy, cc, 'align')
    imagesc(reshape(sta(:,ind(cc)), [NX NY]))
    axis off
end

%%
figure(1); clf
subplot(1,2,1)
imagesc(squeeze(X(i,:,:)))
subplot(1,2,2)
imagesc(squeeze(V(i,:,:)))

%% Get spatiotemporal RFs
NC = size(Robs,2);
nlags = 20;
Rdelta = Robs - mean(Robs);
nstim = size(Stim,2);
ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc < 5.2 & labels == 1;
R = Robs;
Rbar = mean(R);

stas = zeros(nlags, nstim, NC);
for lag = 1:nlags
    fprintf('%d/%d lags\n', lag, nlags)
    ix = ecc < 5.2 & labels == 1;
    ix = find(ix);
    ix = ix(ix > lag);
    Rdelta = R(ix,:) - Rbar;
    stas(lag, :, :) = (X(ix-lag,:))'*Rdelta;
end

%% 
cc = 0;

%% plot one by one
figure(2); clf
cc = cc + 1;
if cc > NC
    cc = 1;
end

sta = stas(:,:,cc);
% zscore the sta
sta = (sta - mean(sta(:))) ./ std(sta(:));

clim = max(abs(sta(:)));
% x = xax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
% y = yax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
xax = (1:NX)/Exp.S.pixPerDeg*60;
yax = (1:NY)/Exp.S.pixPerDeg*60;
for ilag = 1:nlags
   subplot(2,ceil(nlags/2), ilag, 'align')
   imagesc(xax, yax, reshape(sta(ilag,:), [NX NY])', [-1 1]*clim)
end

% colormap(plot.viridis)
colormap(gray)
title(cc)


%%

figure(1); clf
imagesc(reshape(mean(squeeze(std(stas,[], 1)),2), [80, 80]))


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
