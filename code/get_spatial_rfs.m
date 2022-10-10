function RFs = get_spatial_rfs(Exp, varargin)
% get_spatial_rfs(Exp, varargin)
% Get the spatial receptive fields for a population of units
% 
ip = inputParser();
ip.addParameter('ROI', [-15, -10, 15 10])
ip.addParameter('binSize', .5)
ip.addParameter('Frate', 60)
ip.addParameter('win', [-5 20])
ip.addParameter('eyeposExclusion', 16)
ip.addParameter('rfthresh', 0.5)
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
    'smoothing', 2);

% use indices only while eye position is on the screen
scrnBnds = (Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg;
scrnBnds = 1.5 * scrnBnds;
eyePosAtFrame = opts.eyePosAtFrame/Exp.S.pixPerDeg;

ix = (eyePosAtFrame(:,1) + ip.Results.ROI(1)) >= -scrnBnds(1) & ...
    (eyePosAtFrame(:,1) + ip.Results.ROI(3)) <= scrnBnds(1) & ...
    (eyePosAtFrame(:,2) + ip.Results.ROI(2)) >= -scrnBnds(2) & ...
    (eyePosAtFrame(:,2) + ip.Results.ROI(4)) <= scrnBnds(2);

fprintf('%02.2f%% of gaze positions are safely on screen\n', 100*mean(ix))

numspikes = sum(RobsSpace(ix,:));

% forward correlation
stasFull = forwardCorrelation(full(Xstim), RobsSpace-mean(RobsSpace, 1), win, find(ix), [], true, false);

% get summary statistics
RFs = [];
if isempty(RobsSpace)
    return
end
NC = numel(vis);
field = 'Dots';
sig = arrayfun(@(x) x.(field).sig, vis);
mfr = arrayfun(@(x) x.(field).stimFr, vis);
isiV = arrayfun(@(x) x.isiRate, vis);
% fs_stim = round(1/median(diff(opts.frameTimes)));
% assert(abs(fs_stim - Frate) < 1, 'get_spatial_rfs: Frame rate is wrong.')
tax = 1e3*(win(1):win(2))/Frate;

for cc = 1:NC

%     stas = stasFull(:,:,cc);
    rfflat = stasFull(:,:,cc)./sum(Xstim)*Frate; % individual neuron STA

    % reshape into 3D (space x space x time)
    nlags = numel(tax);
    rf = reshape(rfflat, [nlags, opts.dims]); % 3D spatiotemporal tensor
    
    % clip edges
    rf([1 end],:,:) = 0;
    rf(:,[1 end],:) = 0;
    rf(:,:,[1 end]) = 0;

    adiff = abs(rf - median(rf(:))); % positive difference
    mad = median(adiff(:)); % median absolute deviation
    
    adiff = (rf - median(rf(:))) ./ mad; % normalized (units of MAD)
    


    % compute threshold using pre-stimulus lags
    thresh = max(max(max(adiff(tax<=0,:))), .7*max(adiff(:)));
    
    % threshold and find the largest connected component
    bw = bwlabeln(adiff > thresh);
    s = regionprops3(bw);

    if isempty(s)
        disp('No Centroids. skipping.')
        % no connected components found (nothing crossed threshold)
        continue
    end

    % constrain when the peak lag is viable
    peaklagst = interp1(1:nlags, tax, s.Centroid(:,2));
    s = s(peaklagst>=0 & peaklagst <= 250,:);

     if size(s,1) > 2 % no valid clusters or too many valid clusters means it's noise
        disp('Too many centroids. skipping')
        continue
    end
       
    [maxV, bigg] = max(s.Volume);

    ytx = s.Centroid(bigg,:); % location (in space / time)
    peaklagt = interp1(1:nlags, tax, ytx(2)); % peak lag using centroid
    
    peaklagf = max(floor(ytx(2))-1, find(tax==0));
    peaklagc = min(ceil(ytx(2))+1, nlags);
    peaklag = round(ytx(2));
    
    rfflat = reshape(rf, nlags, []);

    [~, spmx] = max(rfflat(peaklag,:));
    [~, spmn] = min(rfflat(peaklag,:));    

    % spatial RF
    srf = squeeze(mean(rf(peaklagf:peaklagc,:,:), 1));
    
    dt = mode(diff(opts.frameTimes));
    Stmp.rfflat = rfflat;
    Stmp.skernel = srf;
    Stmp.spower = reshape(std(rfflat), opts.dims);
    Stmp.tpower = std(rfflat, [], 2);
    Stmp.tkernel = rfflat(:,spmx)/dt;
    Stmp.tkernelout = rfflat(:,spmn);
    Stmp.xax = opts.xax/Exp.S.pixPerDeg;
    Stmp.yax = opts.yax/Exp.S.pixPerDeg;
    Stmp.lags = (win(1):win(2))*dt;
    Stmp.peaklag = peaklag;
    Stmp.peaklagt = peaklagt;
    Stmp.sig = sig(cc);
    Stmp.mfr = mfr(cc);
    Stmp.isiV = isiV(cc);
    Stmp.numspikes = numspikes(cc);
    
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
        continue
    end
        
    % clip edges
    srf = srf(2:end-1,2:end-1);

    [xx,yy] = meshgrid(opts.xax(2:end-1)/Exp.S.pixPerDeg, opts.yax(2:end-1)/Exp.S.pixPerDeg);
%     srf = (srf - min(srf(:))) / (max(srf(:)) - min(srf(:)));
    srf = srf ./ max(srf(:));
    
%     sc = 4;
%     xx = imresize(xx, sc);
%     yy = imresize(yy, sc);
%     srf = imresize(srf, sc);

    [con, ar, ctr, thresh, maxoutrf] = get_rf_contour(xx,yy,srf, 'thresh', rfthresh, 'upsample', 4);
    
    figure(2); clf
    subplot(1,2,1)
    imagesc(xx(1,:), yy(:,1)', srf); hold on
    plot(con(:,1), con(:,2), 'r')

    subplot(1,2,2)
    plot(tax, rfflat(:,spmx)); hold on
    plot(tax, rfflat(:,spmn))
   
    drawnow
    

% 
% 
% %     mask = (hanning(opts.dims(1))*hanning(opts.dims(2))').^.25;
% %     rf = rf .* mask;
% 
%     [xx,yy] = meshgrid(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg);
%     
%     rf = (rf - min(rf(:))) / (max(rf(:)) - min(rf(:)));
%     [con, ar, ctr, maxoutrf] = get_contour(xx,yy,rf, 'thresh', thresh);
    
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
