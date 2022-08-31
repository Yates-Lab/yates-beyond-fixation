function RFs = get_spatial_rfs(Exp)
% get_spatial_rfs
% Get the spatial receptive fields for a population of units

BIGROI = [-30 -10 30 10];
binSize = .5;
Frate = 30;
spike_rate_thresh = 1;
thresh = 4; % z threshold for pixels to count as possibly in the RF

eyePos = Exp.vpx.smo(:,2:3);


dotTrials = io.getValidTrials(Exp, 'Dots');
if ~isempty(dotTrials)
    
    [Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
        'ROI', BIGROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
        'eyePos', eyePos, 'frate', Frate, ...
        'spikeBinSize', 1.1/Frate, ...
        'eyePosExclusion', 1e3, ...
        'fastBinning', true, ...
        'debug', false, ...
        'validTrials', dotTrials);
    
end

% get units to include
spike_rate = mean(RobsSpace)*Frate;

goodunits = find(spike_rate > spike_rate_thresh);

fprintf('%d / %d fire enough spikes to analyze\n', numel(goodunits), size(RobsSpace,2))

% calculate the spatiotemporal RF for each unit using forward correlation
R = RobsSpace(:,goodunits) - mean(RobsSpace(:,goodunits)); % subtract mean
win=[1 10];

% get single temporal kernel for evaluating spatial RF at peak lags
nfolds = 5;
NT = size(R,1);
NC = size(R,2);
xval = regression.xvalidationIdx(NT, nfolds, true);


%% Get single sta for average of all spikes
% ecc = hypot(opts.eyePosAtFrame(:,1), opts.eyePosAtFrame(:,2))/Exp.S.pixPerDeg;
% ecc = hypot(opts.eyePosAtFrame(:,1)-mean(opts.eyePosAtFrame(:,1)), opts.eyePosAtFrame(:,2)-mean(opts.eyePosAtFrame(:,2)))/Exp.S.pixPerDeg;
scrnBnds = (Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg;
scrnBnds = 1.1 * scrnBnds;
eyePosAtFrame = opts.eyePosAtFrame/Exp.S.pixPerDeg;

ix = (eyePosAtFrame(:,1) + BIGROI(1)) >= -scrnBnds(1) & ...
    (eyePosAtFrame(:,1) + BIGROI(3)) <= scrnBnds(1) & ...
    (eyePosAtFrame(:,2) + BIGROI(2)) >= -scrnBnds(2) & ...
    (eyePosAtFrame(:,2) + BIGROI(4)) <= scrnBnds(2);

fprintf('%02.2f%% of gaze positions are safely on screen\n', 100*mean(ix))
valid_inds = find(ix);

stas = forwardCorrelation(Xstim, mean(R,2), win, valid_inds);
stas = stas ./ sum(Xstim);

sd = std(stas(:));
zsta = stas / sd - mean(stas(:));
clim = [-1 1]*max(abs(zsta(:)));
nlags = size(stas,1);
figure(1); clf
for i = 1:nlags
    subplot(2, ceil(nlags/2), i)
    imagesc(reshape(zsta(i,:), opts.dims), clim)
    title(sprintf('lag %d', i))
end

thresh = min(.7*clim(2), thresh);
Tkern = mean(zsta.*(zsta>thresh),2);
Tkern = Tkern / max(Tkern);
Tkern = imboxfilt(Tkern, 3);
figure(2); clf
plot(Tkern, '-o')
title('Temporal Kernel')
drawnow

%% check individual cells
% % ecc = hypot(opts.eyePosAtFrame(:,1)-mean(opts.eyePosAtFrame(:,1)), opts.eyePosAtFrame(:,2)-mean(opts.eyePosAtFrame(:,2)))/Exp.S.pixPerDeg;
% % ix = opts.eyeLabel==1 & ecc < 3.2;
% stas = forwardCorrelation(Xstim, zscore(R), win, find(ix));
% cc = 1;
% %%
% cc = cc + 1;
% if cc > size(stas,3)
%     cc = 1;
% end
% 
% sta = stas(:,:,cc);
% sta = sta ./ sum(Xstim);
% 
% 
% sd = std(sta(:));
% zsta = sta / sd - mean(sta(:));
% clim = [-1 1]*max(abs(zsta(:)));
% nlags = size(stas,1);
% figure(1); clf
% for i = 1:nlags
%     subplot(2, ceil(nlags/2), i)
%     imagesc(reshape(zsta(i,:), opts.dims), clim)
%     title(sprintf('lag %d', i))
% end
% 
% title(cc)
% Tkern = mean(zsta.*(zsta>3.5),2);
% Tkern = Tkern / max(Tkern);
% figure(2); clf
% plot(Tkern, '-o')
% title('Temporal Kernel')
% drawnow


%%
% %% OLD NNMF WAY
% stas = forwardCorrelation(Xstim, R, win, find(ix));
% 
% NC = size(R,2);
% 
% % Get spatial and temporal RFs
% RFspop = zeros([opts.dims NC]);
% nlags = size(stas,1);
% RFtpop = zeros([nlags, NC]);
% 
% for cc = 1:NC
% 
%     I = squeeze(stas(:,:,cc));
%     I = (I - mean(I(:)) )/ std(I(:));
% 
%     [bestlag,~] = find(I==max(I(:)));
%     I = I(min(max(bestlag(1)+[-1 0 1], 1), nlags),:);
%     I = mean(I);
%     I = imgaussfilt(reshape(I, opts.dims), 1);
%     RFspop(:,:,cc) = I;
%     
%     id = find(I(:)==max(I(:)));
%     
%     RFtpop(:,cc) = stas(:,id,cc); %#ok<FNDSB>
%         
% end
% 
% % Do NNMF to find ROI locations
% I = reshape(RFspop, prod(opts.dims), NC);
% 
% 
% score = zeros(4,1);
% for nFac = 1:4
%     rng(1234)
%     [wnmf, hproj] = nnmf(I, nFac);
%     rtmp_ = rsquared(I, wnmf*hproj, false);
%     score(nFac) = mean(rtmp_);
% end
%     
% figure(1); clf
% xax = opts.xax/Exp.S.pixPerDeg;
% yax = opts.yax/Exp.S.pixPerDeg;
% 
% for i = 1:nFac
%     subplot(ceil(nFac/2), 2, i)
%     imagesc(xax, yax, reshape(wnmf(:,i), opts.dims))
%     title(i)
% end
% 
% 
% % user picks factors to use for 
% factors = [1 2 3];
% 
% % project spatial stimuli on nnmf factors
% Xproj = Xstim*wnmf(:,factors);
% 
% nfac = numel(factors);
% 
% % embed time 
% Xd = makeStimRows(Xproj, nlags);
% 
% XX = Xd'*Xd;
% XY = Xd'*R;
% 
% % linear regression
% what = XX \ XY;
% 
% wts = reshape(what, [nlags, nfac, NC]);
% 
% score = zeros(NC, 1);
% for cc = 1:NC
%     score(cc) = rsquared(R(:,cc), Xd*what(:,cc));
% end
% 
% iix = (score > 0.01); % find units that this representation does anything to explain firing rate
% 
% % average those units to get a temporal kernel for the population (this is
% % like analyzing at the best lag, but a weighted average)
% twts = squeeze(mean(wts,2));
% minmax = @(x) (x - min(x)) ./ (max(x) - min(x));
% 
% Tpower = minmax(twts);
% Tkern = mean(Tpower(:,iix),2);
% Tkern = Tkern-min(Tkern);
% plot(Tkern)

%% linear regression on space only

XX = Xstim'*Xstim; % sample covariance
Rfilt = flipud(filter(Tkern, 1, flipud(R))); % spikes at best lag
CpriorInv = qfsmooth(opts.dims(2), opts.dims(1)); % spatial smoothness penalty

lambdas = [100 500 1000 5000 10000];
nlambdas = numel(lambdas);

kmap = zeros([size(Xstim,2), NC, nfolds]);
score = zeros(nlambdas, NC, nfolds);
tic
s = (nfolds-1) / nfolds;
fprintf('Running %d fold cross-validation to find spatial RF\n', nfolds)
for ifold = 1:nfolds
    fprintf('%d / %d folds\n', ifold, nfolds)
    
    
    train_inds = intersect(xval{ifold,1}, valid_inds);
    test_inds = intersect(xval{ifold,2}, valid_inds);
    
    Rtrue = Rfilt(test_inds,:);
    Rtrue = Rtrue./std(Rtrue) - mean(Rtrue); % for zscoring
    
    for ilam = 1:nlambdas
        XY = Xstim(train_inds,:)'*Rfilt(train_inds,:);
        kmap0 = (s*XX + lambdas(ilam)*CpriorInv)\XY;

        mxr2 = max(score(:,:,ifold));

        Rhat = Xstim(test_inds,:)*kmap0;
        Rhat = Rhat./std(Rhat) - mean(Rhat);
        
        score(ilam,:,ifold) = mean(Rtrue .* Rhat); % pearson (ignores scale)

        iix = score(ilam,:,ifold) > mxr2;
        kmap(:,iix,ifold) = kmap0(:,iix);
    end
end

toc

%%
% what = (XX + 1000 * CpriorInv) \ XY;
what = mean(kmap,3);
for cc = 1:NC
    what(:,cc) = mean(kmap(:,cc,max(squeeze(score(:,cc,:))) > 0.05),3);
end
% 
% 
% %%
% what = kmap0;
xax = opts.xax/Exp.S.pixPerDeg;
yax = opts.yax/Exp.S.pixPerDeg;
[xx,yy] = meshgrid(xax, yax);

% get RFs, area, contours
NC = size(Rfilt,2);
ar = zeros(NC,1);
p = zeros(NC,1);

con = cell(NC,1);
ctr = zeros(NC,2);
for cc = 1:NC
    z = reshape(zscore(what(:,cc)), opts.dims);
    p(cc) = sum(z(:)>thresh);
    [con{cc}, ar(cc), ctr(cc,:)] = get_rf_contour(xx, yy, z, 'thresh', thresh, 'plot', false);
    
end


maxscore = max(mean(score,3));

hasrf = find(maxscore > 0.05 & ar' > .6*binSize);
% hasrf = 1:NC;
N = numel(hasrf);
fprintf('%d/%d units have RFs\n', N, NC)

figure(1); clf
plot(hypot(ctr(hasrf,1), ctr(hasrf,2)), sqrt(ar(hasrf)), 'ok', 'MarkerSize', 2)
xlabel('Eccentricity')
ylabel('sqrt(Area)')

sx = round(sqrt(N));
sy = ceil(sqrt(N));

figure(3); clf
set(gcf, 'Color', 'w')
ax = plot.tight_subplot(sx, sy, 0.02, 0.05);

for cc = 1:(sx*sy)
    set(gcf, 'currentaxes', ax(cc))
    if cc > N
        axis off
        continue
    end
        
        
    I = reshape(zscore(what(:,hasrf(cc))), opts.dims);
    
    imagesc(xax, yax, I, [-5 5]); hold on
    plot(con{hasrf(cc)}(:,1), con{hasrf(cc)}(:,2), 'k')
    
    grid on
    
    title(sprintf('%.2f', maxscore(hasrf(cc))))
    title(sprintf('%.2f', ar(hasrf(cc))))
    
end

colormap(plot.coolwarm)

RFs = struct();
RFs.firing_rate_units = goodunits;
RFs.firing_rate_thresh = spike_rate_thresh;
RFs.xax = xax;
RFs.yax = yax;
RFs.lags = (1:nlags) / Frate;
RFs.temporal_kernel = Tkern;
RFs.spatrfs = reshape(what, [opts.dims NC]);
RFs.sig_pixels = p';
RFs.score = maxscore;
RFs.contours = con;
RFs.area = ar';
RFs.center = ctr;
RFs.eccentricity = hypot(ctr(:,1), ctr(:,2));