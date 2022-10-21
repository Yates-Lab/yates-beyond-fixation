
user = 'jakelaptop';
addFreeViewingPaths(user);
addpath scripts/
figDir = 'figures/fig02';

datadir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'preprocessed');
sesslist = arrayfun(@(x) x.name, dir(fullfile(datadir, '*.mat')), 'uni', 0);


%% test session
isess = 1;
sessname = sesslist{isess};
Exp = load(fullfile(datadir, sessname));
wf = io.get_waveform_stats(Exp.osp);
% Get spatial stimulus

%%
dotTrials = io.getValidTrials(Exp, 'Dots');

% BIGROI = [-16 -10 16 10];
% binSize = 1;
BIGROI = [-4 -4 1 1];
binSize = .5;
Frate = 120;
win = [0 15];
eyeposexclusion = 6;
rfthresh = .5;

% check whether firing rates are modulated by stimulus
evalc("vis = io.get_visual_units(Exp, 'plotit', false, 'visStimField', 'BackImage');");

eyePos = Exp.vpx.smo(:,2:3);

[Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
    'ROI', BIGROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
    'eyePosExclusion', inf, ...
    'eyePos', eyePos, 'frate', Frate, ...
    'fastBinning', true, ...
    'smoothing', 1);


%% use indices only there was enough eye positions
bs = 1;
rad = 2;
xax = -10:bs:10;
nx = numel(xax);

n = zeros(nx, nx);
xs = nan(nx, nx);
ys = nan(nx, nx);

eyePosAtFrame = opts.eyePosAtFrame./Exp.S.pixPerDeg;

for i = 1:nx
    for j = 1:nx
        x0 = xax(i);
        y0 = xax(j);
        
        ix = find(hypot(eyePosAtFrame(:,1)-x0, eyePosAtFrame(:,2) - y0) < rad);
        n(j,i) = numel(ix);
        xs(j,i) = x0;
        ys(j,i) = y0;
    end
end

[y0,x0] = find(n==max(n(:)));
x0 = xax(x0);
y0 = xax(y0);

rad = eyeposexclusion;
scrnBnds = (Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg;
scrnBnds = 1.5 * scrnBnds;
eyePosAtFrame = opts.eyePosAtFrame/Exp.S.pixPerDeg;

ix = (eyePosAtFrame(:,1) + BIGROI(1)) >= -scrnBnds(1) & ...
    (eyePosAtFrame(:,1) + BIGROI(3)) <= scrnBnds(1) & ...
    (eyePosAtFrame(:,2) + BIGROI(2)) >= -scrnBnds(2) & ...
    (eyePosAtFrame(:,2) + BIGROI(4)) <= scrnBnds(2);

ix = ix & hypot(eyePosAtFrame(:,1) - x0, eyePosAtFrame(:,2) - y0) < rad;
dist = hypot(opts.eyePosAtFrame(:,1)/Exp.S.pixPerDeg - opts.probex, opts.eyePosAtFrame(:,2)/Exp.S.pixPerDeg - opts.probey);

ix = ix & dist > 0;

fprintf('%02.2f%% of gaze positions are safely on screen\n', 100*mean(ix))

numvalid = sum(ix);
%% run forward correlation

X = Xstim;
Y = RobsSpace; % - filtfilt(ones(600,1)/600, 1, RobsSpace);

% - mean(RobsSpace,1);
inds = find(ix); 
dims = opts.dims;
numlags = 10;
% numlags = win(2);
numboot = 100;

% negative time lags
[stasNull,Nnull] = simpleForcorrValid(X, Y, numboot, inds, -numboot);

% positive time lags
[stasFull, Nstim] = simpleForcorrValid(X, Y, numlags, inds, 0);


%%
NC = size(Y, 2);
ss = std(stasFull);
sn = std(stasNull);
ssrat = squeeze(ss ./ sn);
sn = 1;
[~, iix] = sort(max(ssrat));
% iix = 1:NC;
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(10); clf
ax = plot.tight_subplot(sx, sy, 0, 0);
for i = 1:NC
    set(gcf, 'currentaxes', ax(i));
    cc = iix(i);
    imagesc(reshape(ssrat(:,cc), dims))
    hold on
    plot([0 dims(2)], [0 0], 'w')
    plot([0 dims(2)], [dims(1), dims(1)], 'w')
    plot([dims(2) dims(2)], [0, dims(1)], 'w')
    plot([0 0], [0, dims(1)], 'w')

    axis off
end

figure(12); clf
ax = plot.tight_subplot(sx, sy, 0, 0);
for i = 1:NC
    set(gcf, 'currentaxes', ax(i));
    cc = iix(i);
    imagesc(reshape(ssrat(:,cc), dims), [3 inf]); 
    hold on
    plot([0 dims(2)], [0 0], 'w')
    plot([0 dims(2)], [dims(1), dims(1)], 'w')
    plot([dims(2) dims(2)], [0, dims(1)], 'w')
    plot([0 0], [0, dims(1)], 'w')

    axis off
end

figure(11); clf
ax = plot.tight_subplot(sx, sy, 0, 0);
for i = 1:NC
    set(gcf, 'currentaxes', ax(i));
    cc = iix(i);
    plot(wf(cc).lags, wf(cc).isi)
    axis off
end

%%
%%
if ~exist('cc', 'var')
    cc = 0;
end

cc = cc + 1;
if cc > NC
    cc = 1;
end
% %%

% cc = 38;

sss = zeros(NC,1);
% for cc = 1:NC
figure(6); clf;
null = stasNull(:,:,cc);
sta = stasFull(:,:,cc);

mn = min(null);
mx = max(null);

mu = mean(null);
sdnull = std(null, [], 1);
% figure(13); clf
% subplot(1,2,1)
% imagesc(reshape(sdnull, dims))

p = normcdf((sta - mu) ./ sdnull);
p(p>.5) = 1 - p(p>.5);

alpha = 0.01;
[H, newalpha] = benjaminiHochbergFDR(p(:), alpha);

sigrf = reshape(H, [numlags, prod(dims)]);


subplot(1,3,1)
imagesc(null)

subplot(1,3,2)
imagesc(sta)

subplot(1,3,3)
imagesc( double(sigrf))


ss = (sta - mu) ./ sdnull;
% ss = sta;
mn = min(ss(:));
mx = max(ss(:));
figure(12); clf
for i = 1:numlags
    subplot(2, ceil(numlags/2), i)
    imagesc(reshape(ss(i,:), dims), [mn mx])
end



sigrf = reshape(sigrf, [numlags, dims]); % 3D spatiotemporal tensor
bw = bwlabeln(sigrf);
s = regionprops3(bw);
s
figure(11); clf
for i = 1:numlags
    subplot(2, ceil(numlags/2), i)
    imagesc(squeeze(sigrf(i,:,:)), [0 1])
end
%%
figure(3); clf
subplot(1,2,1)
plot(wf(cc).waveform)
subplot(1,2,2)
plot(wf(cc).lags, wf(cc).isi)
title(cc)
% end
%%
figure(13), clf;
imagesc(reshape(sdnull, dims))
%%

%%

thresh = abs(norminv(0.05/(numlags*prod(dims))));
% mn = min(min(null(:)), min(sta(:)));
% mx = max(max(null(:)), max(sta(:)));
mu = mean(null);
cihigh = mu + thresh*sdnull;
cilow = mu - thresh*sdnull;

subplot(1,3,1)
imagesc(null)
% imagesc(null, [mn mx])
subplot(1,3,2)
imagesc(sta)
% , [mn mx]);
subplot(1,3,3)

sigrf = (sta > cihigh) -  (sta < cilow) ;
imagesc( sigrf)


figure(2); clf
subplot(1,3,1)
sn = reshape(std(stasNull(:,:,cc)), dims);
imagesc(sn)
colorbar
subplot(1,3,2)
ss = reshape(std(stasFull(:,:,cc)), dims);
imagesc(ss)
colorbar
subplot(1,3,3)
imagesc(ss./sn)
title(cc)
colorbar
drawnow
sss(cc) = sum(abs(sigrf(:)));

figure(3); clf
subplot(1,2,1)
plot(wf(cc).waveform)
subplot(1,2,2)
plot(wf(cc).lags, wf(cc).isi)
title(cc)
% end

rf = reshape(sta, [numlags, dims]);
sigrf = reshape(abs(sigrf), [numlags, dims]); % 3D spatiotemporal tensor
bw = bwlabeln(sigrf);
s = regionprops3(bw);
s
%%
[~, bigg] = max([s.Volume]);
s = s(bigg,:);

ytx = s.Centroid; % location (in space / time)
    
peaklag = round(ytx(2));

%%
figure(1); clf

lags = peaklag + (-5:5);
nlags = numel(lags);
for ilag = 1:nlags
    subplot(1,nlags, ilag)
    
    imagesc(squeeze(rf(lags(ilag),:,:))); hold on
end

    

%%

sigrf = reshape(sigrf, [numlags, dims]); % 3D spatiotemporal tensor
sigrf(cc) = mean(sigrf(:));

    bw = bwlabeln(sigrf);
    s = regionprops3(bw);
    if ~isempty(s)
        [~, id] = max(s.Volume);
        s = s(id,:);
        rf.volume(cc) = s.Volume;
        rf.centroids{cc} = s.Centroid;
    end

%%

stat = spat_rf(Exp, 'ROI', BIGROI, ...
    'binSize', .5);

out.coarse = stat;
out.BIGROI = BIGROI;



%% Plot putative ROIs

% include putative significant RFs
six = find(stat.rfsig(:) > 0.001 & stat.maxV(:) > 10 & stat.maxV(:) < 100);

NC = numel(six);

binSize = stat.xax(2)-stat.xax(1);
x0 = min(stat.xax);
y0 = min(stat.yax);


% subplot(1,2,1)
% plot(stat.contours{cc}(:,1), stat.contours{cc}(:,2), 'r')
% xlim(stat.xax([1 end]))
% ylim(stat.yax([1 end]))

mask = zeros(stat.dim);
for i = 1:NC
    cc = six(i);
    
    cx = (stat.contours{cc}(:,1)-x0)/binSize;
    cy = (stat.contours{cc}(:,2)-y0)/binSize;
    mask = mask + poly2mask(cx, cy, size(stat.srf,1), size(stat.srf,2));

end

mask = mask / NC;

figure(2); clf
% , 
imagesc(stat.xax, stat.yax, mask)
hold on

bw = bwlabel(mask > .1);
s = regionprops(mask > .1);
n = max(bw(:));
mx = zeros(n,1);
mn = zeros(n,1);
for g = 1:n
    mx(g) = max(reshape(mask.*(bw==g), [], 1));
    mn(g) = sum(bw(:)==g(:));
end


iix = [s.Area]>2;
if sum(iix) > 0
    s = s(iix);
end

nrois = numel(s);
binsizes = zeros(nrois,1);
rois = cell(nrois,1);
rfstats = cell(nrois,1);

for i = 1:nrois
    
    bb = s(i).BoundingBox;
    bb(1:2) = s.Centroid;
    sz = max(bb(3:4))*[2 2];
    bb(1:2) = bb(1:2) - sz/2;
    bb(3:4) = sz;
%     rectangle('Position', bb , 'EdgeColor', 'r', 'Linewidth', 2)
    NEWROI = [bb(1) bb(2) bb(1) + bb(3) bb(2) + bb(4)];
    ROI = NEWROI*binSize + BIGROI([1 2 1 2]);

    bs = (ROI(3) - ROI(1)) / 20;
    plot(ROI([1 1 3 3 1]), ROI([2 4 4 2 2]), 'r', 'Linewidth', 2)

    binsizes(i) = bs;
    rois{i} = ROI;
end

%% Run analyses at a higher resolution
for i = 1:nrois

    NEWROI = rois{i};
    bs = binsizes(i);
    
    if any(isnan(NEWROI))
        disp('No Valid ROI')
    else
        rfstats{i} = spat_rf(Exp, ...
        'ROI', NEWROI, ...
        'binSize', bs);
    end
end

%%
if numel(rfstats)==1
    id = ones(1, numel(rfstats{1}.r2));
    out.fine = rfstats{1};
    out.NEWROI = rois{1}; % save one for backwards compatibility
else
    [~, id] = max(cell2mat(cellfun(@(x) x.rfsig', rfstats, 'uni', 0)));
    out.fine = rfstats{round(mode(id))};
    out.NEWROI = rois{round(mode(id))}; % save one for backwards compatibility
end

% loop over cells and save the best fit for each one
out.RFs = [];
for cc = 1:numel(id)

    RF = struct();
    RF.timeax = rfstats{id(cc)}.timeax;
    RF.xax = rfstats{id(cc)}.xax;
    RF.yax = rfstats{id(cc)}.yax;
    RF.roi = rfstats{id(cc)}.roi;
    RF.srf = rfstats{id(cc)}.srf(:,:,cc);
    RF.mu = rfstats{id(cc)}.conctr;
    RF.C = rfstats{id(cc)}.rffit(cc).C;
    RF.ar = rfstats{id(cc)}.conarea;
    RF.temporalPref = rfstats{id(cc)}.temporalPref(:,cc);
    RF.temporalNull = rfstats{id(cc)}.temporalNull(:,cc);
    RF.rfsig = rfstats{id(cc)}.rfsig(cc);
    RF.maxV = rfstats{id(cc)}.maxV(cc);
    RF.peaklagt = rfstats{id(cc)}.peaklagt(cc);
    RF.contour = rfstats{id(cc)}.contours{cc};
    RF.nsamples = rfstats{id(cc)}.nsamples;
    RF.frate = rfstats{id(cc)}.frate;
    RF.numspikes = rfstats{id(cc)}.numspikes(cc);

    out.RFs = [out.RFs; RF];
end

%%
figure(1); clf
NC = numel(RFs);
ind = 1:NC;

NC = numel(ind);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

for i = 1:NC
    cc = ind(i);
    subplot(sx, sy, i)
    imagesc(RFs(cc).xax, RFs(cc).yax, RFs(cc).skernel); hold on
    plot(RFs(cc).contour(:,1), RFs(cc).contour(:,2), 'r')
    axis off
end


figure(2); clf
for i = 1:NC
    cc = ind(i);
    subplot(sx, sy, i)
    plot(RFs(cc).tkernel); hold on
    plot(RFs(cc).tkernelout)
    axis off
end




%%

outdir = './output/fig02_spat_rf';
 
set(groot, 'DefaultFigureVisible', 'off')

Srf = cell(numel(sesslist), 1);
overwrite = false;
for isess = 1:numel(sesslist)
    sessname = sesslist{isess};

    fname = fullfile(outdir, sessname);
    
    if exist(fname, 'file') && ~overwrite
        tmp = load(fname);
        RFs = tmp.RFs;

    else
        Exp = load(fullfile(datadir, sessname));

        RFs = get_spatial_rfs(Exp);
        
        save(fname, '-v7.3', 'RFs');
        
    end

    Srf{isess} = RFs;
    
end
disp("Done")
set(groot, 'DefaultFigureVisible', 'on')

%%
RFs = cell2mat(Srf);
sig = arrayfun(@(x) x.rfsig*1e3, RFs);
maxV = arrayfun(@(x) x.maxV, RFs);
maxoutrf = arrayfun(@(x) x.maxoutrf, RFs);
numspikes = arrayfun(@(x) x.numspikes, RFs);

ecc = arrayfun(@(x) hypot(x.ctr(1), x.ctr(2)), RFs);
ar = arrayfun(@(x) x.area, RFs);
arx = arrayfun(@(x) max(x.contour(:,1)) - min(x.contour(:,1)), RFs);
ary = arrayfun(@(x) max(x.contour(:,2)) - min(x.contour(:,2)), RFs);
ar = (arx .* ary)/2;

validwf = numspikes > 300;
six = sig > .1 & maxV > 5 & maxoutrf < .8 & validwf;

figure(1); clf
plot(ecc(six), sqrt(ar(six)), '.'); hold on

plot(xlim, [1 1], 'r')

%%
%% Rosa comparison Eccentricity plot

% get index
fid = 1;
fprintf(fid, '%d/%d (%2.2f%%) units selective for space\n', sum(six), sum(validwf), 100*sum(six)/sum(validwf));

eccx = .1:.1:20; % eccentricity for plotting fits

% Receptive field size defined as sqrt(RF Area)
% rosa_fit = exp( -0.764 + 0.495 * log(eccx) + 0.050 * log(eccx) .^2); % rosa 1997 marmoset
% rosaHat = exp( -0.764 + 0.495 * log(x) + 0.050 * log(x) .^2); % rosa 1997 marmoset
 
figure(2); clf
x = ecc(six);
scaleFactor = sqrt(-log(.5))*2; % scaling to convert from gaussian SD to FWHM
y = ar(six) * scaleFactor;

hPlot = plot(x, y, 'o', 'Color', .8*[1 1 1], 'MarkerFaceColor', .2*[1 1 1], 'MarkerSize', 1.5, 'Linewidth', .25); hold on


b0 = [0.764, 0.495 ,0.050]; % initialize with Rosa fit
fun = @(p,x) exp( -p(1) + p(2)*log(x) + abs(p(3))*log(x).^2);

options = optimset(@lsqcurvefit);
options.Display ='none';
% [bhat,RESNORM,resid,~,~,~,J] = robustlsqcurvefit(fun, b0, x, y, [], [], [], options);
[bhat,RESNORM,resid,~,~,~,J] = lsqcurvefit(fun, b0, x, y, [], [], options);
bhatci = nlparci(bhat, resid, 'jacobian', J);


fprintf(fid, 'Rosa Fit:\n');
fprintf(fid, 'A = %02.2f\n', b0(1));
fprintf(fid, 'B = %02.2f\n', b0(2));
fprintf(fid, 'C = %02.2f\n', b0(3));

fprintf(fid, 'OUR Fit:\n');
fprintf(fid, 'A = %02.2f [%02.2f, %02.2f]\n', bhat(1), bhatci(1,1), bhatci(1,2));
fprintf(fid, 'B = %02.2f [%02.2f, %02.2f]\n', bhat(2), bhatci(2,1), bhatci(2,2));
fprintf(fid, 'C = %02.2f [%02.2f, %02.2f]\n', bhat(3), bhatci(3,1), bhatci(3,2));

hold on
cmap = lines;
[ypred, delta] = nlpredci(fun, eccx, bhat, resid, 'Jacobian', J);
plot.errorbarFill(eccx, ypred, delta, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', 'none', 'FaceAlpha', .25);
hPlot(2) = plot(eccx, fun(bhat,eccx), 'Color', cmap(1,:));
hPlot(3) = plot(eccx, fun(b0,eccx), 'Color', cmap(5,:));

xlim([0 20])
ylim([0 20])

r2rosa = rsquared(y,fun(b0,x));
r2fit = rsquared(y, fun(bhat, x));
fprintf(fid, 'r-squared for fit %02.2f and rosa %02.2f\n', r2fit, r2rosa);

set(gca, 'xscale', 'log', 'yscale', 'log')
xlabel('Eccentricity (d.v.a)')
ylabel('RF size (d.v.a)')
set(gcf, 'Color', 'w')
hLeg = legend(hPlot, {'Data', 'Polynomial Fit', 'Rosa 1997'}, 'Box', 'off');
hLeg.ItemTokenSize = hLeg.ItemTokenSize/6;
pos = hLeg.Position;
hLeg.Position(1) = pos(1) - .1;
hLeg.Position(2) = pos(2) - .05;
hLeg.FontName = 'Helvetica';
hLeg.FontSize = 5;

set(gca, 'XTick', [.1 1 10], 'YTick', [.1 1 10])
xt = get(gca, 'XTick');
set(gca, 'XTickLabel', xt)
yt = get(gca, 'YTick');
set(gca, 'YTickLabel', yt)

text(.25, 5, sprintf('n=%d', sum(six)))

% plot.fixfigure(gcf, 7, [2 2], 'FontName', 'Arial', ...
%     'LineWidth',.5, 'OffsetAxes', false);
plot.formatFig(gcf, [1.75 1.5], 'nature')

%%
isess = 2;
processedFileName = sesslist{isess};
fname = fullfile(datadir, processedFileName);
sessname = strrep(processedFileName, '.mat', '');

    
Exp = load(fname);


%%

RFs = get_spatial_rfs(Exp);

%%
NC = numel(RFs);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

RFvolume = [RFs.maxV]';
RFsigpx  = [RFs.rfsig]'*100;
vis = [RFs.vissig]';
numspikes = [RFs.numspikes]';
maxoutrf = [RFs.maxoutrf]';
% [~, ind] = sort(numspikes);
ind = (1:NC)';
thresh = [RFs.thresh]';
sigix = RFsigpx > .05 & RFvolume > 5 & maxoutrf < .8;

sum(sigix)
%%
varnames = {'Num', 'RFVolume', 'RFsigPx', 'Vis', 'NumSpikes', 'MaxOutRF', 'Thresh', 'Sig'};
table(ind, RFvolume(ind), RFsigpx(ind), vis(ind), numspikes(ind), maxoutrf(ind), thresh(ind), sigix(:), 'VariableNames',varnames)


%%

figure(1); clf
ind = find(six);
[~, ii] = sort(ecc(ind));
ind = ind(ii);

NC = numel(ind);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

for i = 1:NC
    cc = ind(i);
    subplot(sx, sy, i)
    imagesc(RFs(cc).xax, RFs(cc).yax, RFs(cc).skernel); hold on
    plot(RFs(cc).contour(:,1), RFs(cc).contour(:,2), 'r')
    axis off
end


figure(2); clf
for i = 1:NC
    cc = ind(i);
    subplot(sx, sy, i)
    plot(RFs(cc).tkernel); hold on
    plot(RFs(cc).tkernelout)
    axis off
end

%%
i = 1;
cc = ind(i);
figure(1); clf
subplot(1,2,1)
imagesc(RFs(cc).xax, RFs(cc).yax, RFs(cc).skernel); hold on
plot(RFs(cc).contour(:,1), RFs(cc).contour(:,2), 'r')

subplot(1,2,2)
plot(RFs(cc).lags, RFs(cc).tkernel); hold on
plot(RFs(cc).lags, RFs(cc).tkernelout)
% 
% 
% RFvolume = [RFs.maxV]';
% RFsigpx  = [RFs.rfsig]'*100;
% vis = [RFs.vissig]';
% numspikes = [RFs.numspikes]';
% maxoutrf = [RFs.maxoutrf]';
% [~, ind] = sort(numspikes);
% thresh = [RFs.thresh]';
% table(RFvolume(ind), RFsigpx(ind), vis(ind), numspikes(ind), maxoutrf(ind), thresh(ind))
%%
figure(3); clf;
subplot(1,3,1)
histogram([RFs.maxV], linspace(0, 100, 100))
subplot(1,3,2)
histogram([RFs.rfsig], linspace(0, .05, 100))
% sum([RFs.maxV]> 10)
%%

    %% get RFs

ROI = [-1 -1 1 1]*12;
binSize = .25;
Frate = 120;
eyeposexclusion = 20;
win = [1 20];

[Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
    'ROI', ROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
    'eyePosExclusion', eyeposexclusion * Exp.S.pixPerDeg, ...
    'eyePos', Exp.vpx.smo(:,2:3), 'frate', Frate, ...
    'fastBinning', true, ...
    'smoothing', 3);

% use indices only while eye position is on the screen
scrnBnds = (Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg;
scrnBnds = 1.5 * scrnBnds;
eyePosAtFrame = opts.eyePosAtFrame/Exp.S.pixPerDeg;

six = (eyePosAtFrame(:,1) + ROI(1)) >= -scrnBnds(1) & ...
    (eyePosAtFrame(:,1) + ROI(3)) <= scrnBnds(1) & ...
    (eyePosAtFrame(:,2) + ROI(2)) >= -scrnBnds(2) & ...
    (eyePosAtFrame(:,2) + ROI(4)) <= scrnBnds(2);

fprintf('%02.2f%% of gaze positions are safely on screen\n', 100*mean(six))

numspikes = sum(RobsSpace(six,:));

%%
stas = forwardCorrelation(full(Xstim), sum(RobsSpace-mean(RobsSpace, 1),2), win, find(six), [], true, false);

%%
sd = std(stas);
[~, id] = max(sd);
figure(1); clf
plot(stas(:,id));
[~, bestlag] = max(stas(:,id));
rdelta = RobsSpace - mean(RobsSpace,1);
%%
bs = 1;
rad = 2;
xax = -8:bs:8;
nt = size(rdelta,1);
nx = numel(xax);
figure(1); clf
n = zeros(nx, nx);
xi = opts.xax/Exp.S.pixPerDeg;
yi = opts.yax/Exp.S.pixPerDeg;
ctrx = nan(nx, nx);
ctry = nan(nx, nx);
xs = nan(nx, nx);
ys = nan(nx, nx);

for i = 1:nx
    for j = 1:nx
        x0 = xax(i);
        y0 = xax(j);
        
        ix = find(hypot(eyePosAtFrame(:,1)-x0, eyePosAtFrame(:,2) - y0) < rad);
        ix(ix > nt-bestlag) = [];
        n(j,i) = numel(ix);
        if n(j,i) < 500
            continue
        end
        sta = Xstim(ix,:)'*mean(rdelta(ix+bestlag,:),2);
        sta = (sta - min(sta(:))) / (max(sta(:))-min(sta(:)));
        [con, ar, ctr, thresh, maxoutrf] = get_rf_contour(xi,yi,reshape(sta, opts.dims),'thresh', .7, 'upsample', 4, 'plot', false);
        ctrx(j,i) = ctr(1);
        ctry(j,i) = ctr(2);
        xs(j,i) = x0;
        ys(j,i) = y0;

%         subplot(nx, nx, (j-1)*nx + i)
%         imagesc(reshape(sta, opts.dims))
%         axis off
%         axis tight
%         drawnow
    end
    fprintf('%d/%d\n', i, nx)
end

%%
cx0 = ctrx(xs==0 & ys==0);
cy0 = ctry(xs==0 & ys==0);

figure(2); clf
subplot(1,3,1)
imagesc(xax, xax, n)
subplot(1,3,2)
imagesc(xax, xax, ctrx-cx0); colorbar
subplot(1,3,3)
imagesc(xax, xax, ctry-cy0); colorbar

%%
iix = n>500;
Fx = scatteredInterpolant(xs(iix),ys(iix),ctrx(iix)-cx0);
Fy = scatteredInterpolant(xs(iix),ys(iix),ctry(iix)-cy0);

%%
[xx,yy] = meshgrid(linspace(-8,8,100));

figure(3); clf
subplot(1,2,1)
v = Fx(xx,yy);
imagesc(xax, xax, v)
subplot(1,2,2)
v = Fy(xx,yy);
imagesc(xax, xax, v)

%%
eyepos = Exp.vpx.smo(:,2:3);

shiftx = -Fx(eyepos(:,1), eyepos(:,2));
shifty = -Fy(eyepos(:,1), eyepos(:,2));

eyepos(:,1) = eyepos(:,1) + shiftx;
eyepos(:,2) = eyepos(:,2) + shifty;

[Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
    'ROI', ROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
    'eyePosExclusion', eyeposexclusion * Exp.S.pixPerDeg, ...
    'eyePos', eyepos, 'frate', Frate, ...
    'fastBinning', true, ...
    'smoothing', 3);

% use indices only while eye position is on the screen
scrnBnds = (Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg;
scrnBnds = 1.5 * scrnBnds;
eyePosAtFrame = opts.eyePosAtFrame/Exp.S.pixPerDeg;

six = (eyePosAtFrame(:,1) + ROI(1)) >= -scrnBnds(1) & ...
    (eyePosAtFrame(:,1) + ROI(3)) <= scrnBnds(1) & ...
    (eyePosAtFrame(:,2) + ROI(2)) >= -scrnBnds(2) & ...
    (eyePosAtFrame(:,2) + ROI(4)) <= scrnBnds(2);

fprintf('%02.2f%% of gaze positions are safely on screen\n', 100*mean(six))

numspikes = sum(RobsSpace(six,:));

%%
stas = forwardCorrelation(full(Xstim), sum(RobsSpace-mean(RobsSpace, 1),2), win, find(six), [], true, false);

%%
% imagesc(reshape(sta(:,cc)))
clim = [min(stas(:)) max(stas(:))];
figure(1); clf;
nlags = size(stas,1);
nlags = 1;
for ilag = 1
    subplot(2,nlags, ilag)
    imagesc(reshape(stas(bestlag,:), opts.dims), clim)
    subplot(2,nlags, ilag+nlags)
    imagesc(reshape(stas0(bestlag,:), opts.dims), clim)
end

%%
wm = [min(stas(:)) max(stas(:))];
wm = [0 1];

poslags = find(lags > 0);
% poslags = find(lags < 0);
mean(reshape(abs(stas(poslags,:)),[],1)>ci)
nlags = numel(poslags);

for ilag = 1:nlags
    subplot(2, ceil(nlags/2), ilag)
    I = reshape(stas(poslags(ilag), :), opts.dims);
    I = I > ci;
    imagesc(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg, I, wm)
%     title(sprintf('lag: %02.2f', ilag*16))
    title(mean(I(:))*100)
    axis xy
end

    %%
    stas = reshape(stas, [41, 2209, 1]);
    %%
    lags = win(1):win(2);
%     cc = cc + 1;
    cc = 1;
    figure(1); clf
    imagesc(stas(lags < 0,:,cc))
    mu = mean(stas(lags<0,:,cc));
    s = stas(lags > 0, :, cc) - mu;
%     mu = reshape(mu, opts.dims);
    clim = [min(s(:)) max(s(:))];
    figure(2); clf
    
    for ilag = 1:20
        subplot(2, 10, ilag)

        imagesc(reshape(s(ilag, :), opts.dims), clim)
    end
   
   
%%

stas = forwardCorrelation(full(Xstim), RobsSpace-mean(RobsSpace, 1), win, find(six), [], true, false);


%%
close all
figure(1); clf
cc = cc + 1;
if cc > size(stas,3)
    cc = 1;
end

lags = win(1):win(2);

%%
figure(1); clf
for cc = 1:size(stas,3)
mu = mean(stas(lags<0,:,cc));
s = stas(:,:,cc) - mu;

ci = prctile(reshape(abs(s(lags<0,:)), [], 1), 99.9);
plot(mean(s > ci,2)); hold on
end

%%
wm = [min(s(:)) max(s(:))];
% wm = [0 1];

poslags = find(lags > 0);
% poslags = find(lags < 0);
mean(reshape(abs(s(poslags,:)),[],1)>ci)
nlags = numel(poslags);

for ilag = 1:nlags
    subplot(2, ceil(nlags/2), ilag)
    I = reshape(s(poslags(ilag), :), opts.dims);
%     I = I > ci;
    imagesc(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg, I, wm)
%     title(sprintf('lag: %02.2f', ilag*16))
    title(mean(I(:))*100)
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

