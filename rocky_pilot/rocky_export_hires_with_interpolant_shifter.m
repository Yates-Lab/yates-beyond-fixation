
%% Load session
% Exp = load('~/Downloads/r20230608.mat/');
% Exp = load('~/Downloads/r20230618_dpi2.mat/');

load('~/Dropbox/MarmoLabWebsite/PSA/Allen_Reimport/Allen_2022-06-01_13-31-38_V1_64b.mat');




% goodunits = Exp.osp.clusterDepths < 1600;
% cids = Exp.osp.cids(goodunits);

% iix = ismember(Exp.osp.clu, cids);
% fields = fieldnames(Exp.osp);
% newsp = Exp.osp;
% for i = 1:numel(fields)
%     if numel(Exp.osp.(fields{i}))==numel(iix)
%         newsp.(fields{i}) = Exp.osp.(fields{i})(iix);
%     elseif numel(Exp.osp.(fields{i}))==numel(goodunits)
%         newsp.(fields{i}) = Exp.osp.(fields{i})(goodunits);
%     end
% end

figure(1); clf
xax = -15:.1:15;
eyeX = Exp.vpx.smo(:,2);
eyeY = Exp.vpx.smo(:,3);
C = histcounts2(eyeX, eyeY, xax, xax);
center = [-3,1];a

imagesc(xax, xax, C');
hold on
plot(center(1), center(2), 'or', 'MarkerSize', 10)

%%

sessname = 'r20230618';
figname = sprintf('results_%s_4.pdf', sessname);

%% get coarse resolution spatial RFs

ROI = [-1 -1 1 1]*10;
% coarse bin size
binSize = .25;

Frate = 60;
eyeposexclusion = 15;
win = [1 5];


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
RobsSpace = RobsSpace(:,numspikes>1e3);
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

%% pick a lag and get the sta for each neuron
lag = 2;
inds = find(ix);
inds = inds(inds < size(Xstim,1)-lag-1);
sta = 0;
for lag = 2:4
    sta = sta + Xstim(inds,:)'* RobsSpace(inds + lag,:)./sum(RobsSpace(inds + lag,:));
end
% sta = sta-mean(sta)-mean(sta,2);

NC = size(sta,2);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(1); clf
ax = plot.tight_subplot(sx, sy, 0, 0, 0);
zsta = zscore(sta); 
for cc = 1:NC
    set(gcf, 'currentaxes', ax(cc))
    imagesc(reshape(zsta(:,cc), opts.dims), [-5, 5]); 
    axis off
end
% size(stas(:,))
colormap(plot.coolwarm)

%% find ROI
ROIWINDOWSIZE = 200; % spatial dimensions of the high-res ROI
winsize = ROIWINDOWSIZE;
rf = reshape(std(stas), opts.dims);
rf = (rf - min(rf(:))) / (max(rf(:)) - min(rf(:)));
[con, ar, ctr] = get_rf_contour(opts.xax, opts.yax, rf, 'thresh', .7);
% ctr(1) = ctr(1) + 50;
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
saveas(gcf, fullfile('figures/hires_export', sprintf('%s_roi.pdf', sessname)) )
% pixels run down so enforce this here
S.rect([2 4]) = sort(-S.rect([2 4]));

%% Do high-res reconstruction using PTB (has to replay the whole experiment)
Exp.FileTag = sessname;
S.spikeSorting = 'kilo';
%     {'Dots', 'Gabor', 'BackImage', 'Grating', 'FixRsvpStim'}
fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Gabor'}, 'overwrite', true, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false, 'EyeSmoothing', 3, 'EyeSmoothingOrder', 1);

%% test that it worked
% fname = '~/Dropbox/Datasets/Mitchell/stim_movies/r20230608_-50_-50_50_50_1_0_1_19_0_0.hdf5';
% fname = '~/Dropbox/Datasets/Mitchell/stim_movies/r20230608_-71__61_209_341_1_0_1_19_0_0.hdf5';
% fname = '~/Dropbox/Datasets/Mitchell/stim_movies/r20230618_-50_-50_50_50_1_0_1_19_0_0.hdf5';
% fname = '~/Dropbox/Datasets/Mitchell/stim_movies/r20230618_-54_117__96_267_1_0_1_19_0_0.hdf5';
fname = '~/Dropbox/Datasets/Mitchell/stim_movies/r20230618_-79__99_121_299_1_0_1_3_0_0.hdf5';

% fname = '~/Dropbox/Datasets/Mitchell/stim_movies/r20230618_-75_115_125_315_1_0_1_3_0_0.hdf5';
id = 1;
stim = 'Gabor';
tset = 'Train';
%
sz = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'size');
%
iFrame = 1;
%
%% show sample frame

iFrame = iFrame + 1;
I = h5read(fname, ['/' stim '/' tset '/Stim'], [iFrame, 1,1], [1 sz(1:2)']);
I = squeeze(I);
% I = h5read(fname{1}, ['/' stim '/' set '/Stim'], [1,1,iFrame], [sz(1:2)' 1]);
figure(id); clf
imagesc(I)
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
sp = struct();
sp.st = st;
sp.clu = clu;
sp.cids = cids;
Robs = binNeuronSpikeTimesFast(sp, ftoe-8e-3, 1/frate);
% Robs = Robs(:,cids);
% Robs =
eyeAtFrame = h5read(fname, ['/' stim '/' tset '/eyeAtFrame']);
labels = h5read(fname, ['/' stim '/' tset '/labels']);
NX = size(Stim,2);
NY = size(Stim,3);


Stim = reshape(Stim, size(Stim, 1), NX*NY);
Stim = zscore(single(Stim));


%%

figure(1); clf
xax = -15:.1:15;
eyeX = (eyeAtFrame(:,2)-Exp.S.centerPix(1))/Exp.S.pixPerDeg;
eyeY = (eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
C = histcounts2(eyeX, eyeY, xax, xax);
center = [-3,1];

imagesc(xax, xax, C');
hold on
plot(center(1), center(2), 'or', 'MarkerSize', 10)

figure(2); clf
plot(eyeX); hold on
plot(find(labels==1), eyeX(labels==1), '.')
%%
x = (hypot(eyeX, eyeY) - sgolayfilt(hypot(eyeX, eyeY), 1, 3));
figure(1); clf
% 
% labels(x.^2 < 1) = 1;
labels(x.^2 > .05) = 4;
figure(2); clf
plot(eyeX); hold on
plot(find(labels==1), eyeX(labels==1), '.')

%% Pick a lag and compute the STA quickly for all cells


lag = 2;
ecc = hypot(eyeX-center(1), eyeY-center(2));
figure(1); clf
plot(ecc)
hold on
start = lag+1; %ceil(.6*numel(ecc));
ix = find(ecc < 15 & (1:numel(ecc))'> start & labels==1);
plot(ix, ecc(ix), '.')

cids = h5read(fname, ['/Neurons/' spike_sorting '/cids']);
% cids = [653, 753, 834, 968, 972, 1002, 1007, 1022, 1031, 1048, 1067, 1189, 1192, 1194, 1201, 1202, 1279, 1670, 1672, 1972, 1988];
% cids = intersect(cids, find(sum(Robs)>5e3));
% ix = ix(1:50e3);
Rdelta = Robs(:,cids) - mean(Robs(:,cids));

NC = size(Rdelta,2);     
Rdelta = Rdelta(ix,:);
sta = (Stim(ix-lag,:).^2)'*Rdelta;
% sta = (Stim(ix-lag,:))'*Rdelta;
% sta = (Stim(find(ix)-lag,:))'*Rdelta;


ind = 1:NC;
%%
[~, ind] = sort(std(sta));
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(10); clf
ax = plot.tight_subplot(sx, sy, 0, 0, 0);
for cc = 1:NC
%     subplot(sx, sy, cc, 'align')
    set(gcf, 'currentaxes', ax(cc))
    imagesc(reshape(sta(:,ind(cc)), [NX NY]))
    axis off
    title(cids(ind(cc)))
end
colormap(plot.coolwarm)

%%

set(gcf, 'PaperSize', [7 7], 'PaperPosition', [0 0 7 7])
exportgraphics(gcf, figname);

figure(11); clf
plot(std(sta(:,ind)), '-o'); hold on
thresh = 200;
% cids = find(std(sta) > thresh);
plot(xlim, thresh*[1 1], 'r')
set(gcf, 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
exportgraphics(gcf, figname, 'Append', true);

%%
cc = cc + 1;
if cc > NC
    cc = 1;
end

figure(10); clf
imagesc(reshape(sta(:,ind(cc)), [NX NY]))
title(cc)

figure(11); clf
I = reshape(mean(sta,2), [NX, NY]);
I = imgaussfilt(I,3);
imagesc(I)

%% Plot histogram of gaze positions
xgrid = -8:1:8;
xax = xgrid;

eyeX = (eyeAtFrame(:,2)-Exp.S.centerPix(1))/Exp.S.pixPerDeg;
eyeY = (eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;

C = histcounts2(eyeX, eyeY, xax, xax);
figure(1); clf
subplot(1,2,1)
imagesc(xax, xax, C')
title('Eye At Frame')

iix = ~isnan(hypot(Exp.vpx.smo(:,2), Exp.vpx.smo(:,3)));
validTrials = io.getValidTrials(Exp, 'Gabor');
vix = getTimeIdx(Exp.vpx.smo(:,1), cellfun(@(x) x.START_EPHYS, Exp.D(validTrials)), cellfun(@(x) x.END_VPX, Exp.D(validTrials)));
iix = vix & iix;
C2 = histcounts2(Exp.vpx.smo(iix,2), -Exp.vpx.smo(iix,3), xax, xax);
subplot(1,2,2)
imagesc(xax, xax, C2')
title('vpx Smo')
colormap parula

set(gcf, 'PaperSize', [7 7], 'PaperPosition', [0 0 7 7])
exportgraphics(gcf, figname, 'Append', true);

%% Loop over gaze positions and re-calculate Spike-Triggered Energy, find centers
% cids = [411 408];
R = Robs(:,cids);
Rbar = mean(R);

n = numel(xax);
win = 1;

figure(10); clf
set(gcf, 'Color', 'w')
ctrs = nan(n, n, 2);
threshs = nan(n, n);

axs = plot.tight_subplot(n, n, .01, .01);

for ii = 1:n
    for jj = 1:n

        win_ = min(max(hypot(eyeX, eyeY)*win, 1), 3);
        ix = (hypot(eyeX - xax(ii), eyeY - xax(jj)) < win_) & labels == 1;
        fprintf('x: %.2f, y: %0.2f, %d samples\n', xax(ii), xax(jj), sum(ix))
        if sum(ix) < 200
            continue
        end

        ix = find(ix);
        ix = ix(ix > lag);
        Rdelta = R(ix,:) - Rbar;
        sta = (Stim(ix-lag,:).^2)'*Rdelta;
        %         sta = reshape((V(ix-lag,:,:).^2), [],size(Stim,2))'*Rdelta;

        %         subplot(n, n, (ii-1)*n + jj)
        set(gcf, 'currentaxes', axs((jj-1)*n + ii))
        I = reshape(mean(sta,2), [NX, NY]);
        I = imgaussfilt(I,3);
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


% plot.fixfigure(gcf, 12, [7 7])
set(gcf, 'PaperSize', [7 7], 'PaperPosition', [0 0 7 7])
exportgraphics(gcf, figname, 'Append', true);

%% plot map of errors
figure(11); clf
set(gcf, 'Color', 'w')
subplot(1,2,1)
imagesc(ctrs(:,:,1).*(threshs<.6))
title('Horizontal RF center')
xlabel('Gaze Position (d.v.a.)')
ylabel('Gaze Position (d.v.a.)')
subplot(1,2,2)
imagesc(ctrs(:,:,2).*(threshs<.6))
title('Vertical RF center')
xlabel('Gaze Position (d.v.a.)')
ylabel('Gaze Position (d.v.a.)')

set(gcf, 'PaperSize', [7 5], 'PaperPosition', [0 0 7 5])
exportgraphics(gcf, figname, 'Append', true);

%% fit interpolant to errors
xax = xgrid;
[xx, yy] = meshgrid(xax);

figure(1); clf;
[i,j] = find(xx==0 & yy==0);
cx = NX/2;
cy = NY/2;

mask = (threshs<.6);
errX = ctrs(:,:,1)-cx;
errY = ctrs(:,:,2)-cy;
dX = conv2(errX, [-1 1], 'same').^2 + conv2(errX, [-1; 1], 'same').^2; imagesc(dX)
dY = conv2(errX, [-1 1], 'same').^2 + conv2(errX, [-1; 1], 'same').^2; imagesc(dX)
mask = dX < 20 & dY < 20 & mask;
errX = errX.*mask;
errY = errY.*mask;
% errX = ctrs(:,:,1).*mask-cx;
% errY = ctrs(:,:,2).*mask-cy;
errYflip = ctrs(:,:,2).*mask-cy;


plot3(xx(:), yy(:), errX(:), '.'); hold on

% Set up fittype and options.
ft = 'linear'; % interpolant type
options = fitoptions(ft, 'Normalize', 'on');

[xData, yData, zDataX] = prepareSurfaceData( xx(mask), yy(mask), errX(mask) );
[fitresultX, gofX] = fit( [xData, yData], zDataX, ft, fitoptions);

[xData, yData, zDataY] = prepareSurfaceData( xx(mask), yy(mask), errY(mask) );
[fitresultY, gofY] = fit( [xData, yData], zDataY, ft, fitoptions);

[xData, yData, zDataYflip] = prepareSurfaceData( xx(mask), yy(mask), errYflip(mask) );
[fitresultYflip, gofYflip] = fit( [xData, -yData], zDataYflip, ft, fitoptions);

% plot
figure(1); clf
subplot(1,2,1)
h = plot( fitresultX, [xData, yData], zDataX );
legend( h, 'Cubic Interpolation', 'z vs. x, y', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'x', 'Interpreter', 'none' );
ylabel( 'y', 'Interpreter', 'none' );
zlabel( 'z', 'Interpreter', 'none' );
grid on
view( -14.7, 11.1 );

subplot(1,2,2)
h = plot( fitresultY, [xData, yData], zDataY );
legend( h, 'Cubic Interpolation', 'z vs. x, y', 'Location', 'NorthEast', 'Interpreter', 'none' );
% Label axes
xlabel( 'x', 'Interpreter', 'none' );
ylabel( 'y', 'Interpreter', 'none' );
zlabel( 'z', 'Interpreter', 'none' );
grid on
view( -14.7, 11.1 );
%
% figure(2); clf
% h = plot( fitresultYflip, [xData, -yData], zDataYflip );
% legend( h, 'Cubic Interpolation', 'z vs. x, y', 'Location', 'NorthEast', 'Interpreter', 'none' );
% % Label axes
% xlabel( 'x', 'Interpreter', 'none' );
% ylabel( 'y', 'Interpreter', 'none' );
% zlabel( 'z', 'Interpreter', 'none' );
% grid on
% view( -14.7, 11.1 );

%%
xHat = fitresultX(xx, yy);
yHat = fitresultY(xx, yy);
yHatflip = fitresultYflip(xx, -yy);

plot3(xx(:), yy(:), xHat(:), '.')

for i = find(mask)'
    plot3([xx(i) xx(i)], [yy(i) yy(i)], [errX(i) xHat(i)], 'k-'); hold on
end

figure(2); clf,
subplot(1,2,1)
imagesc(errX)
title('Horizontal', 'Measured')
subplot(1,2,2)
imagesc(reshape(xHat, [n n]))
title('Horizontal', 'Interpolant')

set(gcf, 'PaperSize', [7 5], 'PaperPosition', [0 0 7 5])
exportgraphics(gcf, figname, 'Append', true);

figure(3); clf,
subplot(1,2,1)
imagesc(xax, xax, errY)
title('Vertical', 'Measured')
subplot(1,2,2)
imagesc(xax, xax, reshape(yHat, [n n]))
title('Vertical', 'Interpolant')

set(gcf, 'PaperSize', [7 5], 'PaperPosition', [0 0 7 5])
exportgraphics(gcf, figname, 'Append', true);

% figure(3); clf,
% subplot(1,2,1)
% imagesc(reshape(yHat, [n n]), [-20 20])
% subplot(1,2,2)
% imagesc(reshape(yHatflip, [n n]), [-20 20])

%% Do shifting
[xx,yy] = meshgrid(1:NX, 1:NY);

NT = size(Stim, 1);
V = reshape(Stim, [NT, NX, NY]);
shiftX = fitresultX(eyeX, eyeY);
shiftY = fitresultY(eyeX, eyeY);

% 
% %%
% % V = zeros(size(X));

for i = 1:NT
    V(i,:,:) = interp2(squeeze(V(i,:,:)), shiftX(i)+xx, shiftY(i)+yy, 'nearest', 0);
end
disp("Done")

%% check whether shifts produce nans

failedshift = squeeze(sum(sum(isnan(V), 2),3))>0;
shiftmag = hypot(shiftX, shiftY);
eyedist = hypot(eyeX, eyeY);


figure(1); clf
histogram(eyedist(~failedshift), 'BinEdges', linspace(0, 10, 100)); hold on
histogram(eyedist(failedshift), 'BinEdges', linspace(0, 10, 100))
legend({'Successful Shift', 'Failed Shift'})
xlabel('Gaze distance from center')

set(gcf, 'PaperSize', [4 4], 'PaperPosition', [0 0 4 4])
exportgraphics(gcf, figname, 'Append', true);


%%
cids = h5read(fname, ['/Neurons/' spike_sorting '/cids']);
NC = numel(cids);

lag = 4;
ecc = hypot(eyeX-center(1), eyeY-center(2));
figure(1); clf
plot(ecc)
hold on

ix = find(ecc < 10 & (1:numel(ecc))'> lag & labels==1);
% ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
% ix = ecc < 5.2 & labels == 1 & (1:numel(ecc))'> lag;
Rdelta = Robs(:,cids) - mean(Robs(:,cids));
Rdelta = Rdelta(ix,:);
X = reshape(V, size(Stim));
X(isnan(X))=0;
sta0 = (X(ix-lag,:))'*Rdelta;

sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(12); clf
ax = plot.tight_subplot(sx, sy, 0, 0, 0);
for cc = 1:NC
%     subplot(sx, sy, cc, 'align')
    set(gcf, 'currentaxes', ax(cc))
    imagesc(reshape(sta0(:,cc), [NX NY]))
    axis off
    title(cids(cc))
end

set(gcf, 'PaperSize', [7 7], 'PaperPosition', [0 0 7 7])
exportgraphics(gcf, figname, 'Append', true);


%% check that you can create the same shift from the raw eye traces

% shift calculated on this (processed) dataset
shiftX = fitresultX(eyeX, eyeY);
shiftY = fitresultY(eyeX, eyeY);

% shift using "raw" eye pos
shiftX2 = fitresultX(eyeAtFrame(:,5), -eyeAtFrame(:,6));
shiftY2 = fitresultY(eyeAtFrame(:,5), -eyeAtFrame(:,6));

figure(1); clf
xdelta = (shiftX - shiftX2);
ydelta = (shiftY - shiftY2);
mdx = mean(xdelta.^2, 'omitnan');
mdy = mean(ydelta.^2, 'omitnan');

histogram(xdelta, 'BinEdges', linspace(-1, 1, 100))
hold on
histogram(ydelta, 'BinEdges', linspace(-1, 1, 100))
xlabel('Difference in Shifter output (pixels)')

set(gcf, 'PaperSize', [7 7], 'PaperPosition', [0 0 7 7])
exportgraphics(gcf, figname, 'Append', true);

% histogram(xdelta, 'BinEdges', linspace(-10, 10, 100))


%% Regenerate Gabor Stimulus file with shifts

shifter.hshift = fitresultX;
shifter.vshift = fitresultY;
% 'Dots', 'Gabor', 'BackImage', 'Grating', 'FixRsvpStim'
fname2 = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Gabor'}, 'overwrite', false, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false, 'EyeCorrection', shifter, 'EyeSmoothing', 19, 'EyeSmoothingOrder', 1);

%% compute STAs and confirm they have cleaned up
spike_sorting = 'kilo';
Stim = h5read(fname2, ['/' stim '/' tset '/Stim']);
ftoe = h5read(fname2, ['/' stim '/' tset '/frameTimesOe']);

frate = h5readatt(fname2, ['/' stim '/' tset '/Stim'], 'frate');
st = h5read(fname2, ['/Neurons/' spike_sorting '/times']);
clu = h5read(fname2, ['/Neurons/' spike_sorting '/cluster']);
cids = h5read(fname2, ['/Neurons/' spike_sorting '/cids']);
sp = struct();
sp.st = st;
sp.clu = clu;
sp.cids = cids;
Robs = binNeuronSpikeTimesFast(sp, ftoe-8e-3, 1/frate);

eyeAtFrame2 = h5read(fname2, ['/' stim '/' tset '/eyeAtFrame']);
labels = h5read(fname2, ['/' stim '/' tset '/labels']);
NX = size(Stim,2);
NY = size(Stim,3);
NC = size(Robs,2);

% flatten and zscore the stimulus
Stim = reshape(Stim, size(Stim, 1), NX*NY);
Stim = zscore(single(Stim));

% Pick a lag and compute the STA quickly for all cells
lag = 8;
ecc = hypot(eyeAtFrame2(:,2)-Exp.S.centerPix(1), eyeAtFrame2(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
% ix = ecc < 5.2 & labels == 1 & (1:numel(ecc))'> lag;
ix = labels == 1 & (1:numel(ecc))'> lag;
Rdelta = Robs - mean(Robs);
Rdelta = Rdelta(ix,:);
sta = (Stim(find(ix)-lag,:))'*Rdelta;

sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(10); clf
for cc = 1:NC
    subplot(sx, sy, cc, 'align')
    imagesc(reshape(sta(:,cc), [NX NY]))
    axis off
    title(cc)
end

set(gcf, 'PaperSize', [7 7], 'PaperPosition', [0 0 7 7])
exportgraphics(gcf, figname, 'Append', true);


fexport{isess} = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Dots', 'Gabor', 'BackImage', 'Grating', 'FixRsvpStim'}, 'overwrite', false, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false, 'EyeCorrection', shifter, 'EyeSmoothing', 19, 'EyeSmoothingOrder', 1);




%% Get spatiotemporal RFs
NC = size(Robs,2);
nlags = 9;
Rdelta = Robs - mean(Robs);
nstim = size(Stim,2);
% ecc = hypot(eyeAtFrame2(:,2)-Exp.S.centerPix(1), eyeAtFrame2(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc < 5.2 & labels == 1 & (1:numel(ecc))'> lag;
R = Robs;
Rbar = mean(R);

stas = zeros(nlags, nstim, NC);
for lag = 1:nlags
    fprintf('%d/%d lags\n', lag, nlags)
    ix = ecc < 5.2 & labels == 1;
    ix = find(ix);
    ix = ix(ix > lag);
    Rdelta = R(ix,:) - mean(R(ix,:));
    stas(lag, :, :) = (X(ix-lag,:).^2)'*Rdelta;
end

%% 
cc = 0;
cc = 405;

%% plot one by one
figure(2); clf
cc = cc + 1;
if cc > NC
    cc = 1;
end
% cc = 59;
sta = stas(:,:,cc);
% zscore the sta
% sta = (sta - mean(sta(:))) ./ std(sta(:));

clim = max(abs(sta(:)));
clim = max(clim, .1);
% x = xax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
% y = yax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
xax = (1:NX)/Exp.S.pixPerDeg*60;
yax = (1:NY)/Exp.S.pixPerDeg*60;
for ilag = 1:nlags
   subplot(2,ceil(nlags/2), ilag, 'align')
   imagesc(xax, yax, reshape(sta(ilag,:), [NX NY])', [-1 1]*clim)
   title(ilag)
end

% % colormap(plot.viridis)
colormap(parula)
title(cc)
%%
% 
% fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Dots', 'Gabor', 'BackImage', 'Grating', 'FixRsvpStim'}, 'overwrite', true, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false, 'EyeCorrection', shifter);

% fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Dots', 'Gabor', 'BackImage', 'Grating', 'FixRsvpStim'}, 'overwrite', false, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false, 'EyeCorrection', shifter, 'EyeSmoothing', 19, 'EyeSmoothingOrder', 1);
fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'FixRsvpStim'}, 'overwrite', true, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false, 'EyeCorrection', shifter, 'EyeSmoothing', 19, 'EyeSmoothingOrder', 1);

%% Copy to server
server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'; %/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
command = 'scp ';
command = [command fname ' '];
command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname)

