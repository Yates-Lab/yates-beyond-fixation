function RFs = get_spatial_rfs(Exp, varargin)
% get_spatial_rfs(Exp, varargin)
% Get the spatial receptive fields for a population of units
% 
ip = inputParser();
ip.addParameter('ROI', [-15, -10, 15 10])
ip.addParameter('binSize', .5)
ip.addParameter('Frate', 120)
ip.addParameter('win', [0 20])
ip.addParameter('eyeposExclusion', 16)
ip.addParameter('rfthresh', 0.7)
ip.parse(varargin{:});

dotTrials = io.getValidTrials(Exp, 'Dots');
if isempty(dotTrials)
    RFs = [];
    return
end

BIGROI = ip.Results.ROI;
binSize = ip.Results.binSize;
Frate = ip.Results.Frate;
win = ip.Results.win;
eyeposexclusion = ip.Results.eyeposExclusion;
rfthresh = ip.Results.rfthresh;

% check whether firing rates are modulated by stimulus
evalc("vis = io.get_visual_units(Exp, 'plotit', false, 'visStimField', 'BackImage');");

eyePos = Exp.vpx.smo(:,2:3);

[Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
    'ROI', BIGROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
    'eyePosExclusion', eyeposexclusion * Exp.S.pixPerDeg, ...
    'eyePos', eyePos, 'frate', Frate, ...
    'fastBinning', true, ...
    'smoothing', 1);

if isempty(RobsSpace)
    RFs = [];
    return
end

%% use indices only there was enough eye positions
bs = 1;
rad = 2;
xax = -10:bs:10;
nx = numel(xax);
figure(1); clf
n = zeros(nx, nx);
xs = nan(nx, nx);
ys = nan(nx, nx);

eyePosAtFrame = opts.eyePosAtFrame./Exp.S.pixPerDeg;

for i = 1:nx
    for j = 1:nx
        x0 = xax(i);
        y0 = xax(j);
        
        ix = find(hypot(eyePosAtFrame(:,1)-x0, eyePosAtFrame(:,2) - y0) < rad);
        ix(ix > nt-bestlag) = [];
        n(j,i) = numel(ix);
        xs(j,i) = x0;
        ys(j,i) = y0;
    end
end

figure(1); clf;

[y0,x0] = find(n==max(n(:)));
x0 = xax(x0);
y0 = xax(y0);
imagesc(xs(1,:), ys(:,1)', n); hold on
colorbar
plot(x0, y0, 'r+')



%%
rad = 5;
scrnBnds = (Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg;
scrnBnds = 1.5 * scrnBnds;
eyePosAtFrame = opts.eyePosAtFrame/Exp.S.pixPerDeg;

ix = (eyePosAtFrame(:,1) + ip.Results.ROI(1)) >= -scrnBnds(1) & ...
    (eyePosAtFrame(:,1) + ip.Results.ROI(3)) <= scrnBnds(1) & ...
    (eyePosAtFrame(:,2) + ip.Results.ROI(2)) >= -scrnBnds(2) & ...
    (eyePosAtFrame(:,2) + ip.Results.ROI(4)) <= scrnBnds(2);

ix = ix & hypot(eyePosAtFrame(:,1) - x0, eyePosAtFrame(:,2) - y0) < rad;

fprintf('%02.2f%% of gaze positions are safely on screen\n', 100*mean(ix))

numspikes = sum(RobsSpace(ix,:));

%%

stasNull = simpleForcorrValid(Xstim, RobsSpace-mean(RobsSpace,1), win(2)+1, find(ix),-20);
[stasFull,NX] = simpleForcorrValid(Xstim, RobsSpace-mean(RobsSpace,1), win(2)+1, find(ix),0);

%%

figure(1); clf
NX = NX ./ max(NX);
imagesc(reshape(NX, opts.dims))

% %% forward correlation
% stasFull = forwardCorrelation(full(Xstim), RobsSpace-mean(RobsSpace, 1), win, find(ix), [], false, false);
% stasNull = forwardCorrelation(full(Xstim), RobsSpace-mean(RobsSpace, 1), -fliplr(win), find(ix), [], false, false);


%%

ss = squeeze(std(stasFull, [], 1));

NC = numel(vis);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(1); clf
for cc = 1:NC
    subplot(sx, sy, cc)
    imagesc(reshape(ss(:,cc), opts.dims));
end

%% get summary statistics
RFs = [];

NC = numel(vis);
field = 'Dots';
sig = arrayfun(@(x) x.(field).sig, vis);
mfr = arrayfun(@(x) x.(field).stimFr, vis);
isiV = arrayfun(@(x) x.isiRate, vis);

tax = 1e3*(win(1):win(2))/Frate;

for cc = 1:NC

    %%
%     cc = 1;
    %
%     cc = cc + 1;
    
%     stas = stasFull(:,:,cc);

    rfflat = stasFull(:,:,cc)./sum(Xstim)*Frate; % individual neuron STA
    rfflatNull = stasNull(:,:,cc)./sum(Xstim)*Frate; % individual neuron STA
    rfflat = rfflat .* NX;
    rfflatNull = rfflatNull .* NX;

    mu = reshape(mean(rfflatNull), opts.dims);
%     mu = imgaussfilt(mu, 2);
    
    
    figure(1); clf;
    subplot(1,3,1); imagesc(reshape(rfflat(5,:)-0, opts.dims)); colorbar
    subplot(1,3,2); imagesc(reshape(rfflat(5,:)-mu(:)', opts.dims)); colorbar
    subplot(1,3,3); imagesc(mu); colorbar


    figure(3); clf;
    subplot(1,3,1); imagesc(reshape(rfflat(5,:)-0, opts.dims)); colorbar
    subplot(1,3,2); imagesc(reshape(rfflat(5,:)-mu(:)', opts.dims)); colorbar
    subplot(1,3,3); imagesc(reshape(std(rfflatNull), opts.dims)); colorbar
    
    null = rfflatNull - mu(:)'; %reshape(rfflatNull - mu(:)', [], 1);
    
    

%     s = reshape(rfflat(5,:)-mu(:)', opts.dims);
%     imagesc(s < ci(1) | s > ci(2)); colorbar

    normrf = rfflat - mu(:)';

    sd = std(rfflatNull);
    sd = reshape(imgaussfilt(reshape(sd, opts.dims), 3), [], 1)';
    
%     null = null / sd;
%     null = null(:);

    ci = [-4 4];
%     ci = prctile(null, [.1 99.9]);

    figure(2); clf
    plot(max(normrf))

    normrf = normrf ./ sd;
%     rfflatNull = rfflatNull / sd;
    
    % reshape into 3D (space x space x time)
    nlags = numel(tax);
    rf = reshape(normrf, [nlags, opts.dims]); % 3D spatiotemporal tensor
    
    hold on
    plot(max(normrf))
%     % clip edges
%     rf([1 end],:,:) = 0;
%     rf(:,[1 end],:) = 0;
%     rf(:,:,[1 end]) = 0;
    
    
    % threshold and find the largest connected component
    sigrf = rf < ci(1) | rf > ci(2);

    bw = bwlabeln(sigrf);
    s = regionprops3(bw);
    s(s.Volume < 2,:) = [];

    sigrf = mean(sigrf(:));

%%
    if isempty(s)
        disp('No Centroids. skipping.')
        % no connected components found (nothing crossed threshold)
        continue
    end

    %% constrain when the peak lag is viable
    peaklagst = interp1(1:nlags, tax, s.Centroid(:,2));
    s = s(peaklagst>=0 & peaklagst <= 250,:);

%      if size(s,1) > 2 % no valid clusters or too many valid clusters means it's noise
%         disp('Too many centroids. skipping')
%         continue
%     end
       
    [maxV, bigg] = max(s.Volume);

    ytx = s.Centroid(bigg,:); % location (in space / time)
    peaklagt = interp1(1:nlags, tax, ytx(2)); % peak lag using centroid
    
    peaklagf = max(floor(ytx(2))-1, find(tax==0));
    peaklagc = min(ceil(ytx(2))+1, nlags);
    peaklag = round(ytx(2));
    
%     rfflat = reshape(rf, nlags, []);

    [~, spmx] = max(rfflat(peaklag,:));
    [~, spmn] = min(rfflat(peaklag,:));    

    % spatial RF
    srf = squeeze(mean(rf(peaklagf:peaklagc,:,:), 1));
    
    Stmp.rfflat = rfflat;
    Stmp.skernel = srf;
    Stmp.spower = reshape(std(rfflat), opts.dims);
    Stmp.tpower = std(rfflat, [], 2);
    Stmp.tkernel = rfflat(:,spmx);
    Stmp.tkernelout = rfflat(:,spmn);
    Stmp.xax = opts.xax/Exp.S.pixPerDeg;
    Stmp.yax = opts.yax/Exp.S.pixPerDeg;
    Stmp.lags = tax;
    Stmp.peaklag = peaklag;
    Stmp.peaklagt = peaklagt;
    Stmp.vissig = sig(cc);
    Stmp.rfsig = sigrf;
    Stmp.mfr = mfr(cc);
    Stmp.isiV = isiV(cc);
    Stmp.numspikes = numspikes(cc);
    Stmp.maxV = maxV;
    
    Stmp.contour = [nan nan];
    Stmp.contourConv = [nan nan];
    Stmp.thresh = nan;
    Stmp.ctr = [nan nan];
    Stmp.area = nan;
    Stmp.areaConvex = nan;
    Stmp.areaRatio = nan;
    Stmp.maxoutrf = nan;

    if peaklag == 1
        RFs = [RFs; Stmp];
%         continue
    end
        
    % clip edges
%     srf = srf(2:end-1,2:end-1);

%     [xx,yy] = meshgrid(opts.xax(2:end-1)/Exp.S.pixPerDeg, opts.yax(2:end-1)/Exp.S.pixPerDeg);
%     srf = (srf - min(srf(:))) / (max(srf(:)) - min(srf(:)));
%     srf = srf ./ max(srf(:));
    [xx,yy] = meshgrid(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg);
%     srf = reshape(std(rfflat), opts.dims);
    srf = Stmp.skernel;
    if -min(srf(:)) > max(srf(:))
        srf = -srf;
    end

    

    [con, ar, ctr, thresh, maxoutrf] = get_rf_contour(xx,yy,srf, 'thresh', rfthresh, 'upsample', 4);

%     I = srf;% I(abs(srf) < 2) = 0;
%     par0 = [ctr ar, 0, 0];
%     gfit = fit2Dgaussian(xx, yy, I, par0);
% 
%     figure(1); clf;
%     imagesc(xx(1,:), yy(:,1)', I)
%     hold on
%     plot.plotellipse(gfit.mu, gfit.C, 1, 'r', 'Linewidth', 2);

    
    figure(2); clf
    subplot(1,2,1)
    imagesc(xx(1,:), yy(:,1)', srf); hold on
    plot(con(:,1), con(:,2), 'r')

    subplot(1,2,2)
    plot(tax, rfflat(:,spmx)); hold on
    plot(tax, rfflat(:,spmn))
   
    drawnow
    
    if ar == 0
        RFs = [RFs; Stmp];
        % no contour found
%         continue
    end
    

    k = convhull(con(:,1), con(:,2));
    arConv = polyarea(con(k,1), con(k,2));

    
    Stmp.contour = con;
    Stmp.contourConv = [con(k,1), con(k,2)];
    Stmp.thresh = thresh;
    Stmp.ctr = ctr;
    Stmp.area = ar;
    Stmp.areaConvex = arConv;
    Stmp.areaRatio = ar / arConv;
    Stmp.maxoutrf = maxoutrf;


    RFs = [RFs; Stmp];
   
end
