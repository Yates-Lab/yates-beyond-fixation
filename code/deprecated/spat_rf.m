function [stat, opts] = spat_rf(Exp, varargin)
% [stat, opts] = spat_rf(Exp, varargin)
% Calculate spatiotemporal RF using forward correlation (including
% pre-stimulus lags)
% This is like a basic spike-triggered average but it calculates the
% modulation in forward time and using units of spike rate. Therefore, the
% interpretation is what the change in spike rate would be for a flash at
% each location. This the impulse reponse function of the neuron over
% space.

ip = inputParser();
ip.addParameter('ROI', [-15, -10, 15 10])
ip.addParameter('binSize', .5)
ip.addParameter('Frate', 120)
ip.addParameter('win', [0 20])
ip.addParameter('eyeposExclusion', 16)
ip.addParameter('rfthresh', 0.7)
ip.addParameter('fitgauss', true)
ip.addParameter('probeExclusion', 0)
ip.parse(varargin{:});

dotTrials = io.getValidTrials(Exp, 'Dots');
if isempty(dotTrials)
    stat = [];
    opts = [];
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
    'eyePosExclusion', inf, ...
    'eyePos', eyePos, 'frate', Frate, ...
    'fastBinning', true, ...
    'smoothing', 1);

if isempty(RobsSpace)
    stat = [];
    return
end


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

ix = (eyePosAtFrame(:,1) + ip.Results.ROI(1)) >= -scrnBnds(1) & ...
    (eyePosAtFrame(:,1) + ip.Results.ROI(3)) <= scrnBnds(1) & ...
    (eyePosAtFrame(:,2) + ip.Results.ROI(2)) >= -scrnBnds(2) & ...
    (eyePosAtFrame(:,2) + ip.Results.ROI(4)) <= scrnBnds(2);

ix = ix & hypot(eyePosAtFrame(:,1) - x0, eyePosAtFrame(:,2) - y0) < rad;
dist = hypot(opts.eyePosAtFrame(:,1)/Exp.S.pixPerDeg - opts.probex, opts.eyePosAtFrame(:,2)/Exp.S.pixPerDeg - opts.probey);

ix = ix & dist > ip.Results.probeExclusion;

fprintf('%02.2f%% of gaze positions are safely on screen\n', 100*mean(ix))

numspikes = sum(RobsSpace(ix,:));
numvalid = sum(ix);
%% run forward correlation

X = Xstim;
Y = RobsSpace - mean(RobsSpace,1);
inds = find(ix); 
dims = opts.dims;
numlags = win(2);

%%
sigrf = get_rf_sig(Xstim, RobsSpace-mean(RobsSpace,1), numlags, find(ix), opts.dims, 'smoothing', 1, 'plot', true, 'thresh', .1);



%% Quick check whether there is structure in spatial power
ss = squeeze(std(sigrf.stasNorm, [], 1));

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

tax = 1e3*(0:numlags-1)/Frate;

for cc = 1:NC
    
    

    if sigrf.volume(cc) == 0
        disp('No Centroids. skipping.')
        % no connected components found (nothing crossed threshold)
        continue
    end

    %% constrain when the peak lag is viable
    maxV = sigrf.volume(cc);

    ytx = sigrf.centroids{cc}; % location (in space / time)
    peaklagt = interp1(1:numlags, tax, ytx(2)); % peak lag using centroid
    
    peaklagf = max(floor(ytx(2))-1, find(tax==0));
    peaklagc = min(ceil(ytx(2))+1, numlags);
    peaklag = round(ytx(2));
    
    
    rfnorm = reshape(sigrf.stasNorm(:,:,cc), [numlags opts.dims]);
    
    % spatial RF
    srf = squeeze(mean(rfnorm(peaklagf:peaklagc,:,:), 1));
    
    Stmp.rfflat = sigrf.stasRaw(:,:,cc);
    Stmp.cid = cc;
    Stmp.skernel = srf;
    Stmp.spower = reshape(std(sigrf.stasRaw(:,:,cc)), opts.dims);
    Stmp.tpower = std(sigrf.stasRaw(:,:,cc), [], 2);
    Stmp.tkernel = nan*tax;
    Stmp.tkernelout = nan*tax;
    Stmp.xax = opts.xax/Exp.S.pixPerDeg;
    Stmp.yax = opts.yax/Exp.S.pixPerDeg;
    Stmp.lags = tax;
    Stmp.peaklag = peaklag;
    Stmp.peaklagt = peaklagt;
    Stmp.vissig = sig(cc);
    Stmp.rfsig = sigrf.sigrf(cc);
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

    %%
    if peaklag == 1
        RFs = [RFs; Stmp];
        continue
    end
%%        
    [xx,yy] = meshgrid(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg);

    srf = Stmp.skernel;
    if -min(srf(:)) > max(srf(:))
        srf = -srf;
    end

    % get contour
    [con, ar, ctr, thresh, maxoutrf] = get_rf_contour(xx,yy,srf, 'thresh', rfthresh, 'upsample', 4);
    
    % convert contour to pixel coordinates
    bs = xx(1,2)-xx(1,1);
    cx = (-xx(1) + con(:,1))/bs;
    cy = (-yy(1) + con(:,2))/bs;
    rmask = poly2mask(cx, cy, opts.dims(1), opts.dims(2));


    % fit gaussian
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
    plot(con(:,1), con(:,2), 'r'); colorbar

    tpref = Stmp.rfflat*rmask(:) ./ sum(rmask(:));
    tnon = Stmp.rfflat*~rmask(:) ./ sum(~rmask(:));

    subplot(1,2,2);
    imagesc((rmask+.5).*srf)
    subplot(1,2,2)
    plot(tax, tpref); hold on
    plot(tax, tnon)
    
    Stmp.tkernel = tpref;
    Stmp.tkernelout = tpref;

    drawnow
    %%
    if ar == 0
        RFs = [RFs; Stmp];
        % no contour found
        continue
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


%% convert to identical structure to spat_rf_reg for backwards compatibility
stat = struct();

stat.timeax = tax;
numlags = numel(tax);

cids = arrayfun(@(x) x.cid, RFs);

stat.xax = opts.xax/Exp.S.pixPerDeg;
stat.yax = opts.yax/Exp.S.pixPerDeg;
stat.fs_stim = Frate;
stat.dim = opts.dims;
stat.ppd = Exp.S.pixPerDeg;
stat.roi = BIGROI;
% stat.spatrf = zeros([opts.dims NC]);
stat.numspikes = numspikes;
stat.nsamples = size(Xstim,1);
stat.numvalid = numvalid;
stat.frate = Frate;

stat.cgs = Exp.osp.cgs;

dims = opts.dims;
nstim = size(Xstim,2);
stat.rf = zeros(numlags, nstim, NC);

stat.temporalPref = zeros(numlags, NC);
stat.temporalNull = zeros(numlags, NC);
stat.srf = zeros([dims, NC]);

stat.contours = cell(NC,1); %arrayfun(@(x) x.contour, RFs, 'uni', 0);
stat.conarea = nan(NC,1);% arrayfun(@(x) x.area, RFs);
stat.conctr = nan(NC,2); %cell2mat(arrayfun(@(x) x.ctr, RFs, 'uni', 0));
stat.conthresh = nan(NC,1); %arrayfun(@(x) x.thresh, RFs);
stat.conmaxout = nan(NC,1); %arrayfun(@(x) x.maxoutrf, RFs);
stat.conconvarea = nan(NC,1); %arrayfun(@(x) x.areaConvex, RFs);
stat.peaklag = nan(NC,1);
stat.peaklagt = nan(NC,1);
stat.maxV = nan(NC,1);
stat.rfsig = nan(NC,1);
stat.vissig = nan(NC,1);
stat.isiV = nan(NC,1);

[xx,yy] = meshgrid(stat.xax, stat.yax);

for cc = 1:NC
    stat.rffit(cc).warning = 1;
end

for i = 1:numel(cids)
    
    cc = cids(i);
    
    stat.conctr(cc,:) = RFs(i).ctr;
    stat.conarea(cc) = RFs(i).area;
    stat.contours{cc} = RFs(i).contour;
    stat.conthresh(cc) = RFs(i).thresh;
    stat.conmaxout(cc) = RFs(i).maxoutrf;
    stat.conconvarea(cc) = RFs(i).areaConvex;
    
    I = RFs(i).skernel;
    I(abs(I)<2) = 0;
    if ip.Results.fitgauss
        gfit = fit2Dgaussian(xx,yy,I, [RFs(i).ctr, RFs(i).area, 0, 0]);

        stat.rffit(cc).warning = 0;
        stat.rffit(cc).mu = gfit.mu;
        stat.rffit(cc).C = gfit.C;
        stat.rffit(cc).r2 = gfit.r2;
        stat.rffit(cc).ar = gfit.ar;
        stat.rffit(cc).ecc = hypot(stat.rffit(cc).mu(1),stat.rffit(cc).mu(2));
    else
        stat.rffit(cc).mu = RFs(i).ctr;
        stat.rffit(cc).ar = RFs(i).area;
        stat.rffit(cc).ecc = hypot(stat.rffit(cc).mu(1),stat.rffit(cc).mu(2));

    end

    stat.rf(:,:,cc) = RFs(i).rfflat;
    stat.temporalPref(:,cc) = RFs(i).tkernel;
    stat.temporalNull(:,cc) = RFs(i).tkernelout;
    stat.srf(:,:,cc) = RFs(i).skernel;

    stat.peaklag(cc) = RFs(i).peaklag;
    stat.peaklagt(cc) = RFs(i).peaklagt;
    stat.maxV(cc) = RFs(i).maxV;
    stat.rfsig(cc) = RFs(i).rfsig;
    stat.vissig(cc) = RFs(i).vissig;
    stat.isiV(cc) = RFs(i).isiV;
end



