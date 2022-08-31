function [W,S] = get_waveform_stats(osp, varargin)
% get waveform details
% Inputs:
%   osp <struct> the spike struct from getSpikesFromKilo
% Outputs:
%   w <struct array> array of structs with info about each unit
%       cid             Unit Id
%       depth           Depth along probe
%       uQ              Unit Quality Metric (from getSpikesFromKilo)
%       isiV            ISI violation rate
%       x               X position along probe(s) 
%       lags            Time lags for ISI distribution
%       isi             ISI distribution (in excess spikes/sec)
%       isiE            2 x SE on ISI distribution
%       isiL            A crude measure of line noise
%       peaktime        Time of the peak 
%       peakval         Value at peaktime
%       troughtime      Time of the trough
%       troughval       Value at troughtime
%       waveform        Waveform at "spacing" centered on Unit
%       spacing         spacing to calculate waveform

ip = inputParser();
ip.addParameter('binSize', 1e-3)
ip.addParameter('numLags', 200)
ip.addParameter('spacing', [-100 -50 0 50 100])
ip.addParameter('validEpochs', [])
ip.addParameter('debug', false)
ip.parse(varargin{:});

S = io.getUnitLocations(osp);

binSize = ip.Results.binSize;
numLags = ip.Results.numLags;
spacing = ip.Results.spacing;
debug = ip.Results.debug;

NC = numel(osp.cids);

W = repmat(struct('cid', [], ...
    'depth', [], ...
    'uQ', [], ...
    'isiV', [], ...
    'x', [], ...
    'lags', [], ...
    'isi', [], ...
    'isiE', [], ...
    'isiL', [], ...
    'isiRate', [], ...
    'localityIdx', [], ...
    'peaktime', [], ...
    'peakval', [], ...
    'troughtime', [], ...
    'troughval', [], ...
    'waveform', [], ...
    'shiftlags', [], ...
    'shiftwaveform', [], ...
    'BRI', [], ...
    'normrate', [], ...
    'spacing', []), NC, 1);

% bin spike times
binEdges = min(osp.st):binSize:max(osp.st);
if ~isempty(ip.Results.validEpochs)
    nepochs = size(ip.Results.validEpochs,1);
    vix = false(1, numel(binEdges)-1);
    for epoch = 1:nepochs
        vix(binEdges(1:end-1) >= ip.Results.validEpochs(epoch,1) & binEdges(2:end) < ip.Results.validEpochs(epoch,2)) = true;
    end
else
    vix = true(1, numel(binEdges)-1);
end
        
    
for cc = 1:NC
    
    
    % waveform 
    if debug
        fig = figure(111); clf
        fig.Position = [100 100 800 800];
    end
    
    % unit id
    cid = osp.cids(cc);
    
    % interpolate waveform around the center of mass (handles different
    % electrode spacing / centers)
    nsp = numel(spacing);
    nts = size(S.templates, 1);
    wf = zeros(nts, nsp);
    nshanks = numel(unique(S.xcoords));
    
    if S.useWF
        offset = 1;
    else
        offset = 20; % kilo templates are wide so offset
    end

    
    % resample waveform centerd on spacing
    for i = 1:nts
        
        if nshanks > 1
            mask = abs(S.x(cc) - S.xcoords) < mean(diff(unique(S.xcoords)))/2;
        else
            mask = true(numel(S.ycoords),1);
        end
        
        if sum(mask) >= numel(spacing)
            wf(i,:) = interp1(S.ycoords(mask), squeeze(S.templates(i, mask, cc)), S.y(cc)+spacing);
        else
            wf_ = squeeze(S.templates(i, mask, cc));
            wf(i,1:numel(wf_)) = wf_(:);
        end
    end

    % center waveform on trough
    wf = wf(offset:end,:);
    nts = size(wf,1);
    if S.useWF
        ts = osp.WFtax/1e3;
        centeredTimestamps = (-15:30)/30e3;
    else
        ts = (1:nts)/30e3; % NOTE: sampling rate is hard coded here    
        % shift waveform aligned to maximum excursion (peak or trough)
        centeredTimestamps = (-20:40)/30e3;
    end
    
    %--- find peak and trough
    dwdt = diff(wf(:,3));
    
    % get trough
    troughloc = findZeroCrossings(dwdt, 1);
    troughloc = troughloc(ts(troughloc) < .35/1e3); % only troughs before .35 ms after clipping
    tvals = wf(troughloc,3);
    [~, mxid] = min(tvals);
    troughloc = ts(troughloc(mxid));

    % get peak
    peakloc = findZeroCrossings(dwdt, -1);
    peakloc = peakloc(ts(peakloc) < .65/1e3 & ts(peakloc) > -.25/1e3); % only troughs before .35 ms after clipping
    pvals = wf(peakloc,3);
    [~, mxid] = max(pvals);
    peakloc = ts(peakloc(mxid));
    
    if isempty(peakloc)
        peakloc = nan;
    end
    
    if isempty(troughloc)
        troughloc = 0;
    end
    
    % do the centering
    wnew = zeros(numel(centeredTimestamps), size(wf,2));
   
    for isp = 1:size(wf,2)
        wnew(:,isp) = interp1(ts, wf(:,isp), centeredTimestamps+troughloc);
    end
    
    wamp = sqrt(sum(wf.^2));
    locality = wamp(1) / wamp(3);
    
    % interpolate trough / peak with softamx
    softmax = @(x,y,p) x(:)'*(y(:).^p./sum(y(:).^p));
    
    
    winix = abs(ts - troughloc) < .25/1e3;
    
    troughloc = softmax(ts(winix), max(-wf(winix,3),0), 10);
    
    winix = abs(ts - peakloc) < .25/1e3;
    peakloc = softmax(ts(winix), max(wf(winix,3),0), 10);
    
    trough = interp1(ts, wf(:,3), troughloc);
    peak = interp1(ts, wf(:,3), peakloc);

    
    if isempty(peakloc)
        peakloc = nan;
        peak = nan;
    end
    
    if isempty(troughloc)
        troughloc = nan;
        trough = nan;
    end
    
    [~, CtrCh] = max(sum(S.templates(:,:,cc).^2));
    
    % check if waveform exceeds the confidence intervals at start time
    ciHi = mean(S.ciHi(1:5,CtrCh, cc));
    ciLow = mean(S.ciLow(1:5,CtrCh, cc));
    ExtremityCiRatio = [trough./ciLow peak./ciHi];
    

    % fit line from peak to end of waveform
    tax = ts(peakloc<ts)*1e3; % time past peak
    xr = wf(peakloc<ts,3); % waveform
    
    if isempty(xr)
        PeakSlope = nan;
    else
        [~, ind] = min(xr); % only run up to min val after peak
        tax = tax(1:ind);
        xr = xr(1:ind);
        xr = xr - peak;
        t0 = tax(1);
        tax = tax - t0;
        % linear regression
        td = tax(:);
        wls = (td'*td) \ (td'*xr(:));
        PeakSlope = wls;
    end
    
    
    % save out channel-centered waveform
    ctrChWaveform = S.templates(:,CtrCh, cc);
    ctrChWaveformCiHi = S.ciHi(:,CtrCh, cc);
    ctrChWaveformCiLo = S.ciLow(:,CtrCh, cc);
    
    if debug
        subplot(2,3,[1 4], 'align')
        hold off
        t = osp.WFtax;
        numChan = numel(S.ycoords);
        for ch = 1:numChan
            x = [t(:); flipud(t(:))];
            x = x + S.xcoords(ch)/100;
            y = [S.ciLow(:,ch, cc); flipud(S.ciHi(:,ch, cc))];
            y = -y/2;
            y = y + S.ycoords(ch);
            fill(x, y, 'k', 'FaceAlpha', .5); hold on
        end
        axis ij
        ylabel('Depth')
        
        
        for d = 1:numel(spacing)
            plot(t+S.x(cc)/100, S.y(cc) + spacing(d) - wf(:,d), 'b', 'Linewidth', 2)
        end
        axis tight
        
        subplot(2,3,2) % plot clipped waveform
        plot(osp.WFtax, S.templates(:,CtrCh, cc), 'k', 'Linewidth', 2); hold on
        plot(osp.WFtax, S.ciHi(:,CtrCh, cc), 'k--');
        plot(osp.WFtax, S.ciLow(:,CtrCh, cc), 'k--');
        plot(osp.WFtax, wf(:,3), 'r')
        plot(peakloc*1e3, peak, 'ob', 'MarkerFaceColor', 'b')
        plot(troughloc*1e3, trough, 'og', 'MarkerFaceColor', 'g')
        
        
        plot(xlim, ciHi*[1 1], 'r--')
        plot(xlim, ciLow*[1 1], 'r--')
        
        if trough < ciLow || peak > ciHi
           plot(osp.WFtax, wf(:,3), 'g', 'Linewidth', 2)
        end
        
        title(ExtremityCiRatio)
        
        if ~isnan(PeakSlope)
            plot(tax+t0, xr+peak); hold on
            plot(tax+t0, td*wls+peak, 'c', 'Linewidth', 2)
            
            plot(peakloc*[1 1]*1e3, [0 peak], 'b')
            plot(troughloc*[1 1]*1e3, [0 trough], 'r')
        end
        
    end
    
    % autocorrelation
    

    % unit spike times
    sptimes = osp.st(osp.clu==cid);
    
    if ~isempty(ip.Results.validEpochs)
        vixsp = getTimeIdx(sptimes, ip.Results.validEpochs(:,1), ip.Results.validEpochs(:,2));
        sptimes = sptimes(vixsp);        
    end

    
    [spcnt, ~, id] = histcounts(sptimes, binEdges);
    
    
    % get autocorrelation in units of excess firing rate
    lags = -numLags:numLags;
    ix = id + lags;
    ix(any(ix<1 | ix > numel(spcnt),2),:) = [];
    I = spcnt(ix);
    I(:,lags==0)= 0; % remove zero-lag spike
    
    mu = mean(I);
    mu0 = mean(spcnt(vix));
        
    % binomial confidence intervals
    n = size(I,1);
    binoerr = 2*sqrt( (mu - mu.^2)/n);

    xc = (mu - mu0) / binSize; % in excess spikes/sec
    
    % push error through same nonlinearity ( baseline subtract / binsize)
    err = ((mu + binoerr) - mu0)/binSize - xc;
    
    % rate in the refractory period (1ms after spike)
    refrate = xc(lags==1);
    
    % expected rate using shoulders of the autocorrelation as a baseline
    expectedminrate = mean(xc([1 end]));
    
    normrate = mu/mu0;
    
    % fit autocorrelation on basis
    Duration = 200; % ms
    t = ((binSize*1e3):binSize*1e3:Duration)';
    
    numBasis = 14; % should be a function of duration :-/
    shortestLag = 1;
    nlStretch = 1.6; % 2 equals doubling
    B = raised_cosine(t, numBasis, shortestLag, nlStretch);
%     B(1,1) = 1;
    B = B./sum(B,2);
%     B = orth(B); % orthogonalize the basis
    
% plot Basis    
%     figure(1); clf
%     subplot(131)
%     plot(t, B)
%     xlim([1 10])
%     subplot(132)
%     imagesc(B)
%     subplot(133)
%     plot(t, sum(B,2))
%     ylim([0 1.5])
    fitToSpikes = false;
    
    if fitToSpikes
        % build design matrix
        NT = numel(spcnt);
        Xd = conv2(spcnt(:), B, 'full');
        Xd = Xd(1:NT,:);
        Xd = [ones(NT, 1) Xd];
        
        % offset by one bin (so it doesn't perfectly predict itself)
        n = 2;
        y = [spcnt(n:end)'; zeros(n-1,1)];
        
        % index into valid points
        Xd = Xd(vix,:);
        y = y(vix);
        
        disp('fitting post-spike filter')
        
        lambda = 1e-3; % ridge parameter
        C = blkdiag(0, lambda*eye(size(Xd,2)-1));
        w0 = (Xd'*Xd + C)\(Xd'*y); % ridge regression
        
        w = w0;
        hspike = B*w(2:end)+w(1);
        hspike = hspike/binSize;
    else
        %% fit directly to autocorrelation        
        posix = lags > 0;
        x = xc(posix)';
        
        
        w = pinv(B)*x(:);
        hspike = B*w;

    end

    if debug
        subplot(2,3,3, 'align'); hold off
        posix = lags > 0;
        plot(lags(posix)*binSize*1e3, xc(posix)+mu0/binSize)
        hold on
        plot(t, hspike+mu0/binSize)
        xlim([0 50])
    end
    
    if debug
        keyboard
%         input('check')
    end
    
    % smoothed autocorrelation in spike rate
    aclags = lags(posix)*binSize*1e3;
    autocorr = xc(posix)+mu0/binSize;
    hspike = hspike+mu0/binSize;
    
    %%
    % burst-refractoriness index
    BRI = mean(normrate(lags>=1 & lags<=4));

    W(cc).cid = cid;
    W(cc).isiV = osp.isiV(cc);
    W(cc).uQ = osp.uQ(cc);
    W(cc).depth = S.y(cc);
    W(cc).x = S.x(cc);
    W(cc).lags = aclags;
    W(cc).isi = autocorr;
    W(cc).isifit = hspike(1:numel(autocorr));
    W(cc).isiRate = refrate/expectedminrate;
    W(cc).localityIdx = locality;
    W(cc).isiE = err;
    W(cc).isiL = std(detrend(xc(1:floor(numLags/2)))); % look for line noise in ISI:
    W(cc).peaktime = peakloc;
    W(cc).peakval = peak;
    W(cc).troughtime = troughloc;
    W(cc).troughval = trough;
    W(cc).PeakSlope = PeakSlope;
    W(cc).ExtremityCiRatio = ExtremityCiRatio;
    W(cc).waveform = wf;
    W(cc).ctrChWaveform = ctrChWaveform;
    W(cc).ctrChWaveformCiHi = ctrChWaveformCiHi;
    W(cc).ctrChWaveformCiLo = ctrChWaveformCiLo;
    W(cc).wavelags = ts;
    W(cc).spacing = spacing;
    W(cc).shiftlags = centeredTimestamps;
    W(cc).shiftwaveform = wnew;
    W(cc).BRI = BRI;
    W(cc).normrate = normrate;
    W(cc).cg = osp.cgs(cc);
end
