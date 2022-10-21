
% Assuming an Exp is loaded

% we want to be able to do the coarse-to-fine analysis
%% 

BIGROI = [-16 -10 16 10];
binSize = .5;
Frate = 120;
win = [-5 20];
eyeposexclusion = 20; % include all eye positions
rfthresh = .5; % threshold at 50% of max

% check whether firing rates are modulated by stimulus
% evalc("vis = io.get_visual_units(Exp, 'plotit', false, 'visStimField', 'BackImage');");

eyePos = Exp.vpx.smo(:,2:3);

[Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
    'ROI', BIGROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
    'eyePosExclusion', eyeposexclusion * Exp.S.pixPerDeg, ...
    'eyePos', eyePos, 'frate', Frate, ...
    'fastBinning', true, ...
    'smoothing', 2);

% use indices only while eye position is on the screen
scrnBnds = (Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg;
scrnBnds = 1.5 * scrnBnds;
eyePosAtFrame = opts.eyePosAtFrame/Exp.S.pixPerDeg;

ix = (eyePosAtFrame(:,1) + BIGROI(1)) >= -scrnBnds(1) & ...
    (eyePosAtFrame(:,1) + BIGROI(3)) <= scrnBnds(1) & ...
    (eyePosAtFrame(:,2) + BIGROI(2)) >= -scrnBnds(2) & ...
    (eyePosAtFrame(:,2) + BIGROI(4)) <= scrnBnds(2);

fprintf('%02.2f%% of gaze positions are safely on screen\n', 100*mean(ix))


% forward correlation
stasFull = forwardCorrelation(full(Xstim), RobsSpace-mean(RobsSpace, 1), win, find(ix), [], true, false);

%%

% bin coarsely
RFs = get_spatial_rfs(Exp, 'ROI', BIGROI, 'binSize', binSize, 'Frate', Frate);

%%
I = mean(std(stasFull,[],1), 3);
I = std(mean(stasFull,3));
I = reshape(I, opts.dims);
I = (I - min(I(:))) / (max(I(:))- min(I(:)));
bw = (I>.7);
s = regionprops(bw);

nPotential = numel(s);

figure(1); clf
imagesc(I)
hold on
for i = 1:nPotential
    plot(s(i).BoundingBox(1)+[0 0 1 1 1 0]*s(i).BoundingBox(3), s(i).BoundingBox(2)+[0 1 1 1 0 0]*s(i).BoundingBox(3), 'r', 'Linewidth', 2)
    plot(s(i).Centroid(1), s(i).Centroid(2), 'or')

    NEWROI = [s(i).Centroid(1) - s(i).BoundingBox(3) s(i).Centroid(2) - s(i).BoundingBox(4) s(i).Centroid(1) + s(i).BoundingBox(3) s(i).Centroid(2) + s(i).BoundingBox(4)];
    plot(NEWROI([1 1 3 3 1]), NEWROI([2 4 4 2 2]), 'r', 'Linewidth', 2)
    colorbar
end

%%
ROI = NEWROI*binSize + BIGROI([1 2 1 2]);
bs = (ROI(3) - ROI(1)) / 20;

[stat, opts] = spat_rf_reg(Exp, 'ROI', ROI, ...
    'binSize', bs, ...
    'plot', true, ...
    'spikesmooth', 3, ...
    'fitRF', true, ...
    'debug', true, ...
    'numlags', 12, ...
    'r2thresh', 0.001);


%%

[Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
    'ROI', ROI*Exp.S.pixPerDeg, 'binSize', bs*Exp.S.pixPerDeg, ...
    'eyePosExclusion', eyeposexclusion * Exp.S.pixPerDeg, ...
    'eyePos', eyePos, 'frate', Frate, ...
    'fastBinning', true, ...
    'smoothing', 2);

% use indices only while eye position is on the screen
eyePosAtFrame = opts.eyePosAtFrame/Exp.S.pixPerDeg;

ix = (eyePosAtFrame(:,1) + ROI(1)) >= -scrnBnds(1) & ...
    (eyePosAtFrame(:,1) + ROI(3)) <= scrnBnds(1) & ...
    (eyePosAtFrame(:,2) + ROI(2)) >= -scrnBnds(2) & ...
    (eyePosAtFrame(:,2) + ROI(4)) <= scrnBnds(2);

fprintf('%02.2f%% of gaze positions are safely on screen\n', 100*mean(ix))

numspikes = sum(RobsSpace(ix,:));

% forward correlation
stasROI = forwardCorrelation(full(Xstim), RobsSpace-mean(RobsSpace, 1), win, find(ix), [], true, false);


%%

num_lags = 12;
Xd = makeStimRows(Xstim, num_lags);

valid = find(ix);

CpriorInv = qfsmooth3D([num_lags, fliplr(opts.dims)], 1);
CpriorInv = CpriorInv + .5*eye(size(CpriorInv,2));

% --- regularized regression to find RF
lambdas = [1 10 100 1000 5000 10000 50000 100000]; % hyper parameter

nValid = numel(valid);
rng(1234) % fix random seed for reproducibility
test = randsample(valid, floor(nValid/10));
train = setdiff(valid, test);

% use delta spike rate
Rdelta = RobsSpace - mean(RobsSpace);

rtest = Rdelta(test,:);
r0 = mean(rtest);
XX = (Xd(train,:)'*Xd(train,:)); % covariance
XY = (Xd(train,:)'*Rdelta(train,:));

nlam = numel(lambdas);
r2 = zeros(NC, nlam);
ws = zeros(size(XX,1), NC, nlam);
for ilam = 1:nlam
    w = (XX + CpriorInv*lambdas(ilam)) \ XY;
    ws(:,:,ilam) = w;
    rhat = Xd(test,:)*w;
    r2(:,ilam) = 1 - sum((rtest - rhat).^2) ./ sum((rtest).^2);
end

[r2max, id] = max(r2,[],2);
%%
NC = size(RobsSpace,2);
cc = cc + 1;
if cc > NC
    cc = 1;
end
figure(1); clf
subplot(1,2,1)
imagesc(stasROI(:,:,cc))
% imagesc(reshape(std(stasROI(:,:,cc)), opts.dims))
subplot(1,2,2)
w = reshape(ws(:,cc,id(cc)), [num_lags, size(Xstim,2)]);
imagesc(w)
[i,j]=find(w == max(w(:)));
rf = reshape(w(i,:), opts.dims);
rf = (rf - min(rf(:))) / (max(rf(:))-min(rf(:)));

[con, ar, ctr, thresh, maxoutrf] = get_rf_contour(opts.xax, opts.yax, rf, 'upsample', 4, 'thresh', .7);
imagesc(opts.xax, opts.yax, rf)
hold on
plot(con(:,1), con(:,2), 'r')
title(r2max(cc))
colorbar