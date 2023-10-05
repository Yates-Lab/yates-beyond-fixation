function [stimX, Robs, opts] = preprocess_spatialmapping_data(Exp, varargin)
% Preprocess spatial mapping data for analysis (and export to Python)
% Input:
%   Exp [struct]: Marmoview experiment struct
%       osp: this function requires the original spikes from Kilosort
%   
% Optional Arguments (as argument pairs):
%   ROI [1 x 4]:        gaze-centered ROI (in pixels)
%   binSize [1 x 1]:    spatial bin size (in pixels)
%   debug [logical]:    pause on each frame to test it (default: false)
%   spikeBinSize:       size of the spike bins (in seconds)
%   latency:            photodiode measured latency of the monitor (in
%                       seconds)
%   nTimeLags           number of time lags (# of frames)
% 
% Output:
%   stim [NT x NDIMS]
%   Robs [NT x NCells]
%   opts [struct]
%       .frameTimes
%       .xpos
%       .ypos
%       .eyeAtFrame
%
% 
% 2020 jly  wrote it

ip = inputParser();
ip.addParameter('ROI', [-500 -500 500 500]);
ip.addParameter('binSize', ceil(Exp.S.pixPerDeg))
ip.addParameter('debug', false)
ip.addParameter('spikeBinSize', 1/Exp.S.frameRate)
ip.addParameter('latency', 0)
ip.addParameter('eyePosExclusion', 400)
ip.addParameter('verbose', true)
ip.addParameter('eyePos', [])
ip.addParameter('cids', [])
ip.addParameter('frate', 120)
ip.addParameter('validTrials', [])
ip.addParameter('fastBinning', true)
ip.addParameter('smoothing', 0)
ip.parse(varargin{:});

verbose = ip.Results.verbose;

% --- find valid trials
if isempty(ip.Results.validTrials)
    validTrials = intersect(io.getValidTrials(Exp, 'Dots'), io.getValidTrials(Exp, 'Ephys'));
%     if isempty(validTrials)
%         validTrials = intersect(io.getValidTrials(Exp, 'Gabor'), io.getValidTrials(Exp, 'Ephys'));
%     end
else
    validTrials = ip.Results.validTrials;
end

numValidTrials = numel(validTrials);
if numValidTrials==0 % exit
    fprintf('No valid dots trials in [%s]\n', Exp.FileTag)
    stimX = [];
    Robs = [];
    opts.frameTimes = nan;
    opts.xax = nan;
    opts.yax = nan;
    opts.dims = nan;
    opts.xPosition = nan;
    opts.yPosition = nan;
    opts.eyePosAtFrame = nan; % flip Y ahead of time
    opts.validFrames = nan;
    opts.numDots = nan;
    return
end

if verbose
    fprintf('Found %d valid trials\n', numValidTrials)
end

debug   = ip.Results.debug;
ROI     = ip.Results.ROI;
binSize = ip.Results.binSize;
spikeBinSize = ip.Results.spikeBinSize;


% % Eye calibration
% cx = mode(cellfun(@(x) x.c(1), Exp.D(validTrials)));
% cy = mode(cellfun(@(x) x.c(2), Exp.D(validTrials)));
% dx = mode(cellfun(@(x) x.dx, Exp.D(validTrials)));
% dy = mode(cellfun(@(x) x.dy, Exp.D(validTrials)));

% Extract trial-specific values
frameTimes = cellfun(@(x) x.PR.NoiseHistory(:,1), Exp.D(validTrials(:)), 'uni', 0);

xpos = cellfun(@(x) x.PR.NoiseHistory(:,1+(1:x.PR.noiseNum)), Exp.D(validTrials(:)), 'uni', 0);
ypos = cellfun(@(x) x.PR.NoiseHistory(:,x.PR.noiseNum+1+(1:x.PR.noiseNum)), Exp.D(validTrials(:)), 'uni', 0);

probex = cellfun(@(x) x.PR.ProbeHistory(:,1), Exp.D(validTrials(:)), 'uni', 0);
probey = cellfun(@(x) x.PR.ProbeHistory(:,2), Exp.D(validTrials(:)), 'uni', 0);
probeid = cellfun(@(x) x.PR.ProbeHistory(:,3), Exp.D(validTrials(:)), 'uni', 0);

ntdots = cellfun(@(x) size(x,1), xpos);
ntprobe = cellfun(@(x) size(x, 1), probex);
if ~all(ntdots==ntprobe)
    disp('Fixing probe samples')
    ntrials = numel(probex);
    for itrial = 1:ntrials
        nt = numel(frameTimes{itrial});

        tmpx = nan(nt,1);
        tmpy = nan(nt,1);
        tmpid = nan(nt,1);
        tmpx(1:ntprobe(itrial)) = probex{itrial};
        tmpy(1:ntprobe(itrial)) = probey{itrial};
        tmpid(1:ntprobe(itrial)) = probeid{itrial};

        probex{itrial} = tmpx;
        probey{itrial} = tmpy;
        probeid{itrial} = tmpid;
    end
end


% check if two conditions were run a
nd = cellfun(@(x) size(x,2), xpos);
xpos = cell2mat(xpos(nd == max(nd)));
ypos = cell2mat(ypos(nd == max(nd)));
probex = cell2mat(probex(nd == max(nd)));
probey = cell2mat(probey(nd == max(nd)));
probeid = cell2mat(probeid(nd == max(nd)));

frameTimes = Exp.ptb2Ephys(cell2mat(frameTimes(nd==max(nd))));
frameTimes = frameTimes + ip.Results.latency;

% bin spikes
Robs = binNeuronSpikeTimesFast(Exp.osp, frameTimes, spikeBinSize);
if isempty(ip.Results.cids)
    cids = Exp.osp.cids;
else
    cids = ip.Results.cids;
end
if isempty(intersect(unique(Exp.osp.clu),  Exp.osp.cids))
    warning('provided cids does not match unit ids')
    cids = unique(Exp.osp.clu);
end

Robs = Robs(:,cids);
NX = size(xpos,2);

eyeDat = Exp.vpx.smo(:,1:3);

% convert to pixels
if isempty(ip.Results.eyePos)
    eyeDat(:,2:3) = eyeDat(:,2:3)*Exp.S.pixPerDeg;
else
    eyeDat(:,2:3) = ip.Results.eyePos*Exp.S.pixPerDeg;
end

% convert time to ephys units
eyeDat(:,1) = Exp.vpx2ephys(eyeDat(:,1));

% find index into frames
[~, ~,id] = histcounts(frameTimes, eyeDat(:,1));
if any(id==0)
    id((id==0)) = max(id);
end
eyeAtFrame = eyeDat(id,2:3);
eyeLabels = Exp.vpx.Labels(id);
eyeSpeed = (Exp.vpx.smo(id,7) + Exp.vpx.smo(id-1,7))/2;
    
if debug
    figure(1); clf
    subplot(1,2,1)
    plot(eyeAtFrame(:,1), eyeAtFrame(:,2), '.')
    subplot(1,2,2)
    plot(xpos, ypos, '.')
    
    figure(2); clf
    plot(eyeDat(:,1), eyeDat(:,2), '-o', 'MarkerSize', 2); hold on
    plot(frameTimes, xpos(:,1), '.')
    drawnow
    pause
end

xPosition = xpos - eyeAtFrame(:,1);
% yPosition = -ypos + eyeAtFrame(:,2);
yPosition = ypos - eyeAtFrame(:,2);

valid = hypot(eyeAtFrame(:,1), eyeAtFrame(:,2)) < ip.Results.eyePosExclusion;

% build spatial grid
xax = ROI(1):binSize:ROI(3);
yax = ROI(2):binSize:ROI(4);


%% try faster binning
if verbose
    disp('Binning stimulus on grid')
    tic
end


if ip.Results.fastBinning
    
    
    xp = xPosition(valid,:);
    yp = yPosition(valid,:);
    
%     iframe = 100;
%     figure(1); clf
%     subplot(1,3,1)
%     plot(xp(iframe,:), yp(iframe,:), 'o')

    % pad by bin size for dots
%     sc = .7;
%     xp = [xp, xp + binSize/sc, xp - binSize/sc, xp, xp];
%     yp = [yp, yp, yp, yp+binSize/sc, yp - binSize/sc];

%     subplot(1,3,2)
%     plot(xp(iframe,:), yp(iframe,:), 'o')
%     hold on
%     plot(ROI([1 1]), ROI([2 4]), 'r')
%     plot(ROI([1 3]), ROI([2 2]), 'r')
%     plot(ROI([1 3]), ROI([4 4]), 'r')
%     plot(ROI([3 3]), ROI([2 4]), 'r')
    
    NT = sum(valid);
    fr = repmat((1:NT)', 1,size(xp,2));
    
    xp(xp < ROI(1) | xp > ROI(3)) = nan;
    yp(yp < ROI(2) | yp > ROI(4)) = nan;

%     plot(xp(iframe,:), yp(iframe,:), 'o')
    
    x = ceil(xp./binSize);
    y = ceil(yp./binSize);
    
    x = x - min(x(:));
    y = y - min(y(:));

%     subplot(1,3,3)
%     plot(x(iframe,:), y(iframe,:), 'o')
%     xlim([0 round((ROI(3)-ROI(1))/binSize)])
%     ylim([0 round((ROI(4)-ROI(2))/binSize)])
    
    good = ~(isnan(x) | isnan(y));
    good = good & x < numel(xax) & y<numel(yax) & x > 0 & y > 0;
    
    x = x(good);
    y = y(good);
    
    fr = fr(good);
    
    v = ones(size(fr));
    
    sz = [max(y(:)) max(x(:))];
    ind = sub2ind(sz, y(:), x(:));
    stimX = sparse(fr(:), ind, v(:), NT, prod(sz));

    xax = xax(2:sz(2)+1); %edges
    yax = yax(2:sz(1)+1);  %+binSize/2;
    dims = sz;

    if ip.Results.smoothing > 0
        [xx,yy] = meshgrid(1:dims(2), 1:dims(1));
        I = sparse(exp( -.5*hypot(xx(:)-xx(:)', yy(:)-yy(:)').^4/ip.Results.smoothing^2 ));
        stimX = stimX * I;
    end

else
    [xx,yy] = meshgrid(xax, yax);
    
    % bin stimulus on grid
    dims = [numel(yax) numel(xax)];
    stimX = zeros(sum(valid), prod(dims));
    tic
    if verbose
        disp('Slow binning stimulus on grid')
    end
    
    bs = binSize.^2;
    
    for i = 1:NX
%         stimX = stimX + double(hypot(xPosition(valid,i) - xx(:)', yPosition(valid,i) - yy(:)') < binSize);
        xdelta = hypot(xPosition(valid,i) - xx(:)', yPosition(valid,i) - yy(:)');

        stimX = stimX + exp( -xdelta.^2/2/bs );
        toc
    end
end

if verbose
    fprintf('Done [%02.2f]\n', toc)
end

stimX = full(stimX);

% %
% figure(2); clf, 
% plot(sum(stimX)); hold on; plot(xlim, .5*median(sum(stimX))*[1 1])
% 
% figure(3); clf
% imagesc(reshape(sum(stimX), sz))
% 
% figure(4); clf
% imagesc(reshape(sum(stimX), dims))
% %%

Robs = Robs(valid,:);
frate = Exp.S.frameRate;
t_downsample = ceil(frate / ip.Results.frate);
if t_downsample > 1
    fprintf('Downsampling time by %d\n', t_downsample)
	stimX = downsample_time(stimX, t_downsample) / t_downsample;
	Robs = downsample_time(Robs, t_downsample);
    frameTimes = downsample_time(frameTimes(valid), t_downsample) / t_downsample;
    xpos =  downsample_time(xpos(valid,:), t_downsample) / t_downsample;
    ypos =  downsample_time(ypos(valid,:), t_downsample) / t_downsample;
    probex =  downsample_time(probex(valid,:), t_downsample) / t_downsample;
    probey =  downsample_time(probey(valid,:), t_downsample) / t_downsample;
    probeid =  downsample_time(probeid(valid,:), t_downsample) / t_downsample;
    eyeAtFrame = downsample_time(eyeAtFrame(valid,:), t_downsample) / t_downsample;
    eyeLabels = downsample_time(eyeLabels(valid), t_downsample) / t_downsample;
    eyeSpeed = downsample_time(eyeSpeed(valid), t_downsample) / t_downsample;
    valid =  downsample_time(valid(valid), t_downsample) / t_downsample;
else % apply valid
    frameTimes = frameTimes(valid);
    xpos =  xpos(valid,:);
    ypos =  ypos(valid,:);
    probex =  probex(valid,:);
    probey =  probey(valid,:);
    probeid =  probeid(valid,:);
    eyeAtFrame = eyeAtFrame(valid,:);
    eyeLabels = eyeLabels(valid);
    eyeSpeed = eyeSpeed(valid);
    valid = valid(valid);
end
    
opts.frameTimes = frameTimes;
opts.eyeLabel = eyeLabels;
opts.xax = xax;
opts.yax = yax;
opts.dims = dims;
opts.xPosition = xpos;
opts.yPosition = -ypos;
opts.probex = probex;
opts.probey = probey;
opts.probeid = probeid;
opts.eyePosAtFrame = eyeAtFrame.*[1 -1]; % flip Y ahead of time
opts.eyeSpeed = eyeSpeed;
opts.validFrames = valid;
opts.numDots = size(xpos,2);