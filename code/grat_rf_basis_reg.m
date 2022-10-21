function stat = grat_rf_basis_reg(Exp, varargin)
% Get Grating Receptive Field using forward correlation
% stat = grat_rf_helper(Exp, varargin)
% Inputs:
%   Exp <struct>    main experiment struct
%   
% Optional:
%   'win'         [-5 25]
%   'plot'        true
%   'fitminmax'   false
%   'stat'        <struct>      use existing forward correlation and refit
%                               gaussian
%   'debug'       <logical>     enter debugger to pause/evaluate each unit
%   'boxfilt'     <odd integer> if > 1, filter RF with spatio temporal boxcar
%                               before thresholding
%   'sftuning'    <string>      which parameterization of the spatial
%                               frequency tuning curve to use
%                               ('raisedcosine', 'loggauss')
%   'upsample'    <integer>     upsample spatial RF before fitting (acts as
%                               regularization (default: 1 = no upsampling)
%   'vm01'        <logical>     normalize the von mises between 0 and 1
%                               (default: true -- will automatically relax 
%                               if real bandwidth is too wide)
%   

ip = inputParser();
ip.addParameter('win', [-5 15])
ip.addParameter('plot', true)
ip.addParameter('fitminmax', false)
ip.addParameter('stat', [])
ip.addParameter('debug', false)
ip.addParameter('boxfilt', 1, @(x) mod(x,2)~=0);
ip.addParameter('numBasisOri', 8)
% ip.addParameter('numBasisSf', 4)
ip.addParameter('sftuning', 'loggauss', @(x) ismember(x, {'loggauss', 'raisedcosine'}))
ip.addParameter('upsample', 1)
ip.addParameter('vm01', true)
ip.parse(varargin{:});

switch ip.Results.sftuning
    case 'loggauss'
        log_gauss = 1;
    case 'raisedcosine'
        log_gauss=0;
end


if ~isempty(ip.Results.stat)
    stat = ip.Results.stat;

else
    
    win = ip.Results.win;
    
    fs_stim = 120;
    nlags = 10;

    % --- load stimulus
    validTrials = io.getValidTrials(Exp, 'Grating');
    
    
    eyeDat = Exp.vpx.smo(:,1:3); % eye position
    % convert time to ephys units
    eyeDat(:,1) = Exp.vpx2ephys(eyeDat(:,1));
    
    % edge-case: sometimes high-res eye data wasn't collected for all trials
    % exclude trials without high-res eye data
    tstarts = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
    goodix = tstarts > eyeDat(1,1) & tstarts < eyeDat(end,1);
    
    validTrials = validTrials(goodix);
    
    frameTime = cell2mat(cellfun(@(x) Exp.ptb2Ephys(x.PR.NoiseHistory(:,1)), Exp.D(validTrials), 'uni', 0));
    ori = cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,2), Exp.D(validTrials), 'uni', 0));
    cpd = cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,3), Exp.D(validTrials), 'uni', 0));
    
    cpdmin = min(cpd(cpd>0));
    cpdmax = max(cpd);
    npow = 1.8;
    
    numBasisSf = ceil((log10(cpdmax) - log10(cpdmin)) / log10(npow));
    
    stat.basisendpoints = [cpdmin npow];
    % find index into frames
    [~, ~,id] = histcounts(frameTime, eyeDat(:,1));
    % eyeAtFrame = eyeDat(id,2:3);
    eyeLabels = Exp.vpx.Labels(id);
    
    % evaluate orientation and spatial frequency on polar basis
    Stim = polar_basis_cos(ori, cpd, ip.Results.numBasisOri, numBasisSf, stat.basisendpoints);
    
    binsize = median(diff(frameTime));
    
    Robs = binNeuronSpikeTimesFast(Exp.osp, frameTime, binsize);
    Robs = Robs(:,Exp.osp.cids);
    
    valid = eyeLabels==1;
    
    t_downsample = ceil(Exp.S.frameRate / fs_stim);
    if t_downsample > 1
        Stim = downsample_time(Stim, t_downsample) / t_downsample;
        Robs = downsample_time(Robs, t_downsample) / t_downsample;
        valid = downsample_time(valid, t_downsample) / t_downsample;
    end
    
    valid = find(valid);
    
    NC = size(Robs,2);
    
    % time-embedded stimulus
    Xstim = makeStimRows(Stim, nlags);
    numsamples = size(Xstim,1);
    
    % --- Ridge regression to find RF
    lambdas = [.1 1 10 100 1000 10000 100000]; % ridge parameter
    
    nValid = numel(valid);
    test = randsample(valid, floor(nValid/5));
    train = setdiff(valid, test);
    
    % use delta spike rate
    numspikes = sum(Robs);
    Rdelta = Robs - mean(Robs);
    
    rtest = Rdelta(test,:);
    r0 = mean(rtest);
    XX = (Xstim(train,:)'*Xstim(train,:)); % covariance
    XY = (Xstim(train,:)'*Rdelta(train,:));
    I = eye(size(XX,2));
    
    nlam = numel(lambdas);
    r2 = zeros(NC, nlam);
    ws = zeros(size(XX,1), NC, nlam);
    for ilam = 1:nlam
        w = (XX + I*lambdas(ilam)) \ XY;
        ws(:,:,ilam) = w;
        rhat = Xstim(test,:)*w;
        r2(:,ilam) = 1 - sum((rtest - rhat).^2) ./ sum((rtest-r0).^2);
    end
    
    
    
    % figure(1); clf
    % imagesc(r2); hold on
    % plot(id, 1:NC, 'ko')
    %%
    % get best regularization
    [rmax, id] = max(r2,[],2);
    
    if ip.Results.plot
        figure(1); clf
        sx = ceil(sqrt(NC));
        sy = round(sqrt(NC));
    end
    
    % basis
    [xx, yy] = meshgrid(0:180, 0:.1:max(cpd));
    Basis = polar_basis_cos(xx(:), yy(:), ip.Results.numBasisOri, numBasisSf, stat.basisendpoints);
    dims = size(xx);
    
    nbasis = ip.Results.numBasisOri * numBasisSf;
    mrf = zeros(nlags, nbasis, NC);
    temporalPref = zeros(diff(win)+1, NC);
    temporalPrefSd = zeros(diff(win)+1, NC);
    temporalNull = zeros(diff(win)+1, NC);
    temporalNullSd = zeros(diff(win)+1, NC);
    peaklag = zeros(NC,1);
    srf = zeros([dims, NC]);
    for cc = 1:NC
        mrf(:,:,cc) = reshape(ws(:,cc,id(cc)), [nlags nbasis]);
        
        % get peak lag
        w = mrf(:,:,cc);
        [pl,~] = find(w==max(w(:)),1);

        % --- get temporal rf using forward correlation
        
        % mask out preferred stimulus
        mask = w(pl,:) ./ max(w(pl,:));
        maskPref = mask==1;
        maskNull = mask==min(mask(:));
        
        % preferred stim
        X = Stim*maskPref(:);
        ind = find(X==max(X(:)));
        vind = intersect(ind, valid);
        [an, sd] = eventTriggeredAverage(Rdelta(:,cc), vind, win);
        sd = sd./sqrt(numel(vind));
        
        temporalPref(:,cc) = an*fs_stim;
        temporalPrefSd(:,cc) = sd*fs_stim;
        
        % preferred stim
        X = Stim*maskNull(:);
        ind = find(X==max(X(:)));
        vind = intersect(ind, valid);
        [an, sd] = eventTriggeredAverage(Rdelta(:,cc), vind, win);
        sd = sd./sqrt(numel(vind));
        
        temporalNull(:,cc) = an*fs_stim;
        temporalNullSd(:,cc) = sd*fs_stim;
        
        if ip.Results.plot
            subplot(sx, sy, cc)
            plot.errorbarFill(1:(diff(win)+1), temporalPref(:,cc), temporalPrefSd(:,cc), 'b', 'FaceColor', 'b'); hold on
            plot.errorbarFill(1:(diff(win)+1), temporalNull(:,cc), temporalNullSd(:,cc), 'r', 'FaceColor', 'r'); hold on
        end
        
        srf(:,:,cc) = reshape(Basis*w(pl,:)', dims);
    end
     
    
    if ip.Results.plot
        cmap = plot.coolwarm(40);
    
        figure(2); clf
        for cc = 1:NC
            subplot(sx, sy, cc)
            plot.polar_contour(xx,yy,srf(:,:,cc));
        end
        colormap(cmap)
    end
    %%
    
    % plot basis    
%     figure(1); clf
%     for i = 1:size(Basis,2)
%         subplot(ip.Results.numBasisOri, numBasisSf, i)
%         imagesc(reshape(Basis(:,i), dims))
%     end
    

    % store everything
    stat.rf = mrf;
    stat.timeax = 1e3*(win(1):win(2))/fs_stim;
    stat.dim = [ip.Results.numBasisOri numBasisSf];
    stat.xax = xx;
    stat.yax = yy;
    stat.Basis = Basis;
    stat.Bdims = dims;
    stat.srf = srf;
    
    stat.temporalPref = temporalPref;
    stat.temporalPrefSd = temporalPrefSd;
    stat.temporalNull = temporalNull;
    stat.temporalNullSd = temporalNullSd;
    stat.temporalPref = temporalPref;
    
    stat.peaklag = zeros(NC,1);
    stat.sdbase = zeros(NC,1);
    stat.r2 = rmax; %zeros(NC,1);
    stat.r2rf = rmax;
    stat.lambdamax = lambdas(id)';
    stat.numsamples = numsamples;
    stat.frate = fs_stim;
    stat.numspikes = numspikes;
    
    
    
end
    
    
% loop over and do fitting
NC = size(Robs,2);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

if ip.Results.plot
    figure(3); clf
end

% initialize variables
stat.peaklagt = nan(NC,1);
stat.thresh = nan(NC,1);
stat.maxV = zeros(NC,1);

stat.rffit = repmat(struct('pHat', [], ...
    'fitfun', 'polarRF', ...
    'oriPref', [], ...
    'oriBandwidth', [], ...
    'sfPref', [], ...
    'sfBandwidth', [], ...
    'base', [], ...
    'amp', [], ...
    'r2', [], ...
    'srfHat', [], ...
    'cid', []), NC, 1);

stat.sftuning = ip.Results.sftuning;

% Main Loop over cells:
% 2) Fit parametric model initialized on centroid
for cc = 1:NC
    fprintf('Fitting unit %d/%d...\n', cc, NC)
    
    % initialize the von-mises to be normalized. If the bandwidth is too
    % wide, then relax this constraint
    vm01 = ip.Results.vm01; % is the von-mises bound between 0 and 1 (messes up bandwidth calculation)
    
   
%     if stat.r2rf(cc) < 0.001
%         fprintf('Bad Unit. Skipping\n')
%         continue
%     end
    
    
    I = squeeze(stat.srf(:,:,cc));
    
    [y0,x0]=find(I==max(I(:)));
    if numel(x0)>1
        x0 = x0(end);
        y0 = y0(end);
    end
    % initialize mean for fit based on centroid
%     x0 = interp1(1:numel(stat.xax(1,:)), stat.xax(1,:), x0);
    y0 = interp1(1:numel(stat.yax(:,1)), stat.yax(:,1), y0);
    
    % --- fit parametric RF
    mxI = max(I(:)); %#ok<*NASGU>
    mnI = min(I(:));
    
    fun = @(params, xy) prf.polarRF(xy, params(1), params(2), params(3), params(4), params(5), params(6));
    evalc("phat = lsqcurvefit(fun, [mxI, x0, y0, .1, .5, mnI], [stat.xax(:), stat.yax(:)], I(:));");
    Ifit = reshape(fun(phat, [stat.xax(:), stat.yax(:)]), size(stat.xax));
    
    if ip.Results.plot
        subplot(sx, sy, cc)
        plot.polar_contour(stat.xax, stat.yax, Ifit);
        title(cc)
        colormap(cmap)
    end
        
    if ip.Results.debug
        
        figure(4); clf
        subplot(1,2,1)
        plot.polar_contour(stat.xax, stat.yax, I);
        title('RF')
        subplot(1,2,2)
        plot.polar_contour(stat.xax, stat.yax, Ifit);
        title('Fit')
        colormap(cmap)
        keyboard
    end
    
    r2 = rsquared(I(:), Ifit(:));
    
    val = 1/sqrt(2);
    % orientation bandwidth
    oriBW = acos(log(val)*phat(4) + 1)/pi*180;
    % spatial frequency bandwidth
    x1 = 10*exp(log(2) * -sqrt( -log(val)*phat(5) ) + log(phat(3)/10));
    x2 = 10*exp(log(2) * sqrt( -log(val)*phat(5) ) + log(phat(3)/10));
    sfBW = x2 - x1;
    
    % save 
    stat.rffit(cc).pHat = phat;
    stat.rffit(cc).oriPref = wrapTo180(phat(2));
    stat.rffit(cc).oriBandwidth = oriBW;
    stat.rffit(cc).sfPref = phat(3);
    stat.rffit(cc).sfBandwidth = sfBW;
    stat.rffit(cc).base = phat(6);
    stat.rffit(cc).amp = phat(1);
    stat.rffit(cc).r2 = r2;
    stat.rffit(cc).srfHat = Ifit;
    stat.rffit(cc).cid = cc;

    stat.r2rf(cc) = r2;
    
end

% check significance
stat.sig = stat.r2rf(:) > 0.001 & [stat.rffit.r2]'>.4;
    