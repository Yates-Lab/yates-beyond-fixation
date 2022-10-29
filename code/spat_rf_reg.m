function [stat, opts] = spat_rf_reg(Exp, varargin)
% [stat, opts] = spat_rf_reg(Exp, varargin)
% Calculate spatiotemporal RF using linear regression
% This is like a basic spike-triggered average but it calculates the
% modulation in forward time and using units of spike rate. Therefore, the
% interpretation is what the change in spike rate would be for a flash at
% each location. This the impulse reponse function of the neuron over
% space.
%
% Inputs:
%   Exp <struct> Main experiment structure
% Optional:
%   'ROI',       [-14 -10 14 10]
%   'binSize',   1
%   'win',       [-5 15]
%   'numspace',  20
%   'plot',      true
% Output:
%   stat <struct>
%   opts <struct>

% build default options
ip = inputParser();
ip.addParameter('ROI', [-14 -10 14 10])
ip.addParameter('binSize', 1)
ip.addParameter('win', [-5 15])
ip.addParameter('numspace', 20)
ip.addParameter('numlags', 3)
ip.addParameter('plot', true)
ip.addParameter('frate', 120)
ip.addParameter('spikesmooth', 3)
ip.addParameter('stat', [])
ip.addParameter('fitRF', true)
ip.addParameter('debug', false)
ip.addParameter('r2thresh', 0.01)
ip.parse(varargin{:})

%% build stimulus matrix for spatial mapping
num_lags = ip.Results.numlags;

if ~isempty(ip.Results.stat)
    % existing STA was passed in. re-do neuron stats, but don't recompute
    % forward correlation
    stat = ip.Results.stat;
    NC = numel(stat.thresh);
    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));
    nlags = numel(stat.timeax);
    
else
    eyePos = Exp.vpx.smo(:,2:3);
    
    [Stim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
        'ROI', ip.Results.ROI*Exp.S.pixPerDeg, 'binSize', ip.Results.binSize*Exp.S.pixPerDeg, ...
        'eyePos', eyePos, 'frate', ip.Results.frate);
    
    numspikes = sum(RobsSpace);

    smwin = ip.Results.spikesmooth - mod(ip.Results.spikesmooth, 2) + 1;
    if smwin > 0
        RobsSpace = imboxfilt(RobsSpace, [smwin 1]);
    end
        
    win = ip.Results.win;
    
    NC = size(RobsSpace,2);
    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));
    
%     fs_stim = round(1/median(diff(opts.frameTimes)));
    fs_stim = 1/ip.Results.frate;
    tax = 1e3*(win(1):win(2))/fs_stim;
    
    stat.timeax = tax;
    stat.xax = opts.xax/Exp.S.pixPerDeg;
    stat.yax = opts.yax/Exp.S.pixPerDeg;
    stat.fs_stim = fs_stim;
    stat.dim = opts.dims;
    stat.ppd = Exp.S.pixPerDeg;
    stat.roi = ip.Results.ROI;
    stat.spatrf = zeros([opts.dims NC]);
    stat.numspikes = numspikes;
    stat.nsamples = size(Stim,1);
    stat.frate = ip.Results.frate;
    
    stat.cgs = Exp.osp.cgs(:);
end

for cc = 1:NC
    stat.rffit(cc).warning = 1;
end

%% Build design matrix
Xstim = makeStimRows(Stim, num_lags);


% use indices only while eye position is on the screen and fixating
eyePosAtFrame = opts.eyePosAtFrame/Exp.S.pixPerDeg;
% use indices only while eye position is on the screen
scrnBnds = (Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg;
scrnBnds = 1.5 * scrnBnds;

ix = (eyePosAtFrame(:,1) + ip.Results.ROI(1)) >= -scrnBnds(1) & ...
    (eyePosAtFrame(:,1) + ip.Results.ROI(3)) <= scrnBnds(1) & ...
    (eyePosAtFrame(:,2) + ip.Results.ROI(2)) >= -scrnBnds(2) & ...
    (eyePosAtFrame(:,2) + ip.Results.ROI(4)) <= scrnBnds(2);

fprintf('%02.2f%% of gaze positions are safely on screen\n', 100*mean(ix))

numspikes = sum(RobsSpace(ix,:));

valid = find(opts.eyeLabel==1 & ix);

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
XX = (Xstim(train,:)'*Xstim(train,:)); % covariance
XY = (Xstim(train,:)'*Rdelta(train,:));

nlam = numel(lambdas);
r2 = zeros(NC, nlam);
ws = zeros(size(XX,1), NC, nlam);
for ilam = 1:nlam
    w = (XX + CpriorInv*lambdas(ilam)) \ XY;
    ws(:,:,ilam) = w;
    rhat = Xstim(test,:)*w;
    r2(:,ilam) = 1 - sum((rtest - rhat).^2) ./ sum((rtest-r0).^2);
end

% get best regularization
[rmax, id] = max(r2,[],2);
id = min(id+2, size(r2,2));



dims = opts.dims;
nstim = size(Stim,2);
mrf = zeros(num_lags, nstim, NC);
temporalPref = zeros(diff(win)+1, NC);
temporalPrefSd = zeros(diff(win)+1, NC);
temporalNull = zeros(diff(win)+1, NC);
temporalNullSd = zeros(diff(win)+1, NC);
peaklag = zeros(NC,1);
srf = zeros([dims, NC]);
contours = cell(NC,1);
conarea = nan(NC,1);
conctr = nan(NC,2);
conthresh = nan(NC,1);
conmaxout = nan(NC,1);
conconvarea = nan(NC,1);

if ip.Results.plot
    fig = figure(1); clf
    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));
end

for cc = 1:NC
    mrf(:,:,cc) = reshape(ws(:,cc,id(cc)), [num_lags nstim]);
    
    % get peak lag
    w = mrf(:,:,cc);

    rf = reshape(w, [num_lags, dims]);

    adiff = abs(rf - median(rf(:))); % positive difference
    mad = median(adiff(:)); % median absolute deviation
    
    adiff = (rf - median(rf(:))) ./ mad; % normalized (units of MAD)
    
    % compute threshold using pre-stimulus lags
    thresh = max(max(max(adiff(1,:,:))), .7*max(adiff(:)));
    
    % threshold and find the largest connected component
    bw = bwlabeln(adiff > thresh);
    s = regionprops3(bw);

    if ~isempty(s)
        % constrain when the peak lag is viable
        [~, ind] = sort(s.Volume, 'descend');
        pl = round(s.Centroid(ind(1),2));
        plags = min(max(1, pl + [-1 0 1]), num_lags);


        % --- get temporal rf using forward correlation

        % get RF contour, center, area
        rf = reshape(mean(w(plags,:)), dims);

    else
       
        [pl,~] = find(w == max(w(:)));
        rf = reshape(w(max(pl),:),dims);

    end

%     if numel(s) > 3
%         disp('Too many Centroids. skipping.')
%         % no connected components found (nothing crossed threshold)
%         continue
%     end

    
%     figure(2)
%     for ilag = 1:num_lags
%         subplot(2,ceil(num_lags/2), ilag)
%         imagesc(reshape(w(ilag,:), dims), [min(w(:)), max(w(:))])
%     end
try
    [con, ar, ctr, thresh, maxoutrf] = get_rf_contour(stat.xax, stat.yax, rf, 'upsample', 4, 'thresh', .7, 'plot', false);
    contours{cc} = con;
    conarea(cc) = ar;
    conctr(cc,:) = ctr;
    conthresh(cc) = thresh;
    conmaxout(cc) = maxoutrf;
    
    k = convhull(con(:,1), con(:,2));
    arConv = polyarea(con(k,1), con(k,2));
    conconvarea(cc) = arConv;

    % mask in RF and out of RF
    maskPref = poly2mask(interp1(stat.xax, 1:numel(stat.xax), con(:,1)), interp1(stat.yax, 1:numel(stat.yax), con(:,2)), size(rf,1), size(rf,2));
    maskNull = ~maskPref;
catch
    pl = pl(end);
    mask = (w(pl,:) - min(w(pl,:))) / (max(w(pl,:)) - min(w(pl,:)));
    maskPref = mask>.7;
    maskNull = mask<.3;
end
    
    % preferred stim
    X = Stim*maskPref(:);
    ind = find(diff(X)>0);
    vind = intersect(ind, valid);
    [an, sd] = eventTriggeredAverage(Rdelta(:,cc), vind, win);
    sd = sd./sqrt(numel(vind));
    
    temporalPref(:,cc) = an*fs_stim;
    temporalPrefSd(:,cc) = sd*fs_stim;
    
    % preferred stim
    X = Stim*maskNull(:);
    ind = find(diff(X)>0);
    vind = intersect(ind, valid);
    [an, sd] = eventTriggeredAverage(Rdelta(:,cc), vind, win);
    sd = sd./sqrt(numel(vind));
    
    temporalNull(:,cc) = an*fs_stim;
    temporalNullSd(:,cc) = sd*fs_stim;
    
    if ip.Results.plot
        figure(fig)
        subplot(sx, sy, cc)
        plot.errorbarFill(1:(diff(win)+1), temporalPref(:,cc), temporalPrefSd(:,cc), 'b', 'FaceColor', 'b'); hold on
        plot.errorbarFill(1:(diff(win)+1), temporalNull(:,cc), temporalNullSd(:,cc), 'r', 'FaceColor', 'r'); hold on
    end
    
    srf(:,:,cc) = reshape(w(pl,:)', dims);
end

% store everything
stat.rf = mrf;
stat.timeax = 1e3*(win(1):win(2))/fs_stim;
stat.srf = srf;

stat.temporalPref = temporalPref;
stat.temporalPrefSd = temporalPrefSd;
stat.temporalNull = temporalNull;
stat.temporalNullSd = temporalNullSd;
stat.temporalPref = temporalPref;

stat.peaklag = zeros(NC,1);
stat.sdbase = zeros(NC,1);
stat.r2 = rmax;
stat.r2rf = nan*rmax;
stat.lambdamax = lambdas(id)';

% save contours
stat.contours = contours;
stat.conarea = conarea;
stat.conctr = conctr;
stat.conthresh = conthresh;
stat.conmaxout = conmaxout;
stat.conconvarea = conconvarea;

for cc = 1:NC
    stat.rffit(cc).warning = 1;
end

%% compute cell-by cell quantities / plot (if true)

% if ip.Results.plot
%     fig = figure; clf
% end

%%
if ip.Results.plot
    figure(2); clf
end

% build input for Gaussian fit
[xx,yy] = meshgrid(stat.xax,  stat.yax);
X = [xx(:) yy(:)];

rfLocations = nan(NC,2);

dprime = (stat.temporalPref-stat.temporalNull) ./ (stat.temporalPrefSd/2+stat.temporalNullSd/2);
dpfit = max(dprime(stat.timeax>0,:)) > 2*max(dprime(stat.timeax<0,:));
fitRF = rmax > ip.Results.r2thresh | dpfit;

binsize = mean(diff(stat.xax));
stat.sig = false(NC,1);

for cc = 1:NC
    I = squeeze(srf(:,:,cc));
    I = (I - min(I(:))) ./ (max(I(:))-min(I(:)));
    
    Is = imgaussfilt(I, 1);
%     [y0,x0] = find(Is==max(Is(:)));
    [y0,x0] = radialcenter(Is);
    
    y0 = interp1(1:numel(opts.yax), stat.yax, y0);
    x0 = interp1(1:numel(opts.xax), stat.xax, x0);
    
    if hypot(x0 - mean(stat.xax), y0 - mean(stat.yax)) > 3*binsize
        x0 = mean(stat.xax);
        y0 = mean(stat.yax);
    end
    
    if ip.Results.plot
        subplot(sx, sy, cc)
        imagesc(stat.xax, stat.yax, I); hold on
        axis xy
    end
    
    mxI = max(I(:));
    mnI = min(I(:));
    
    if fitRF(cc) && ip.Results.fitRF
        
        % initial parameter guess
        par0 = [x0 y0 max(2,hypot(x0, y0)*.5) 0 mean(I(:))];
        
        % gaussian function
        gfun = @(params, X) params(5) + (mxI - params(5)) * exp(-.5 * sum(((X-[params(1) params(2)])*pinv([params(3) params(4); params(4) params(3)]'*[params(3) params(4); params(4) params(3)])).*(X-[params(1) params(2)]),2));
        
        % bounds
        lb = [min(xx(:)) min(yy(:)) 0 -1 mnI]; %#ok<*NASGU>
        ub = [max(xx(:)) max(yy(:)) 10 1 mnI+1];
        
        % least-squares
        options = optimoptions('lsqcurvefit', 'Display', 'none');
        
        try
            phat = lsqcurvefit(gfun, par0, X, Is(:), lb, ub, options);
        catch
            phat = par0;
        end
        
        try
            phat = lsqcurvefit(gfun, phat, X, I(:), lb, ub, options);
        end
        
%         [phat,R,~,COVB] = nlinfit(X, I(:), gfun, par0);
%         CI = nlparci(phat, R, 'covar', COVB);
        
        % convert paramters
        mu = phat(1:2);
        C = [phat(3) phat(4); phat(4) phat(3)]'*[phat(3) phat(4); phat(4) phat(3)];
        
        
        % get r2
        Ihat = gfun(phat, X);
        r2 = rsquared(I(:), Ihat(:));
        stat.r2rf(cc) = r2;

        % convert multivariate gaussian to ellipse
        [~,s,~] = svd(C);
        s = sqrt(s);
        ar = pi*prod(diag(s)); % get area
        ecc = hypot(mu(1), mu(2));
        
        resids = abs(Ihat(:) - I(:));
        b0 = sum(resids>.5);
        c = cond(C);
        if c > 15, c = 1; end
        ws = max(diff(stat.xax([1 end])), diff(stat.yax([1 end])));
        mushift = sqrt( sum((mu - [x0 y0]).^2));
        stat.sig(cc) = ar > binsize^2 & c > 1 & ar < ws & b0 < 2 & mushift < 1.25;
        
        
        if ip.Results.plot && stat.sig(cc)
            plot.plotellipse(mu, C, 1, 'r', 'Linewidth', 2);
        end
        fprintf('%d) ecc: %02.2f, area: %02.2f, r^2:%02.2f\n', cc, ecc, ar, r2)
        
        stat.rffit(cc).amp = mxI;
        stat.rffit(cc).C = C;
        stat.rffit(cc).mu = mu;
        stat.rffit(cc).r2 = r2;
        stat.rffit(cc).ar = ar;
        stat.rffit(cc).ecc = ecc;
        stat.rffit(cc).beta = phat;
        stat.rffit(cc).resids = resids;
%         stat.rffit(cc).betaCi = CI;
        stat.rffit(cc).mushift = mushift;
        
        rfLocations(cc,:) = mu;
    else
        stat.rffit(cc).mushift = nan;
        stat.rffit(cc).ecc = nan;
        if ip.Results.fitRF
            fprintf('%d) No RF. skipping\n', cc)
        end
        
    end
    
    
end

stat.rfLocations = rfLocations;

% ms = [stat.rffit.mushift]./[stat.rffit.ecc];
%  ms(:) < .25 & fitRF(:);

