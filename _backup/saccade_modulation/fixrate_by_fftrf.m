function [fftrf, plotopts] = fixrate_by_fftrf(Exp, srf, grf, varargin)
% S = fixrate_by_fftrf(Exp, srf, grf, varargin)
% Calculate stimulus modulation using the spatial RF and grating RF
% measured for each cell applied to the image at each fixation
% Inputs:
%   Exp <struct>    experiment struct
%   srf <struct>    spatial RF struct (output of spat_rf_helper)
%   grf <struct>    grating RF struct (output of grat_rf_helper)
% Optional (as argument pairs):
%   'plot'          plot analysis results along the way? (default: true)
%   'win'           time window start and end in seconds (default = [-.1 .5])
%   'binsize'       spike rate bin size (in seconds)
%   'smoothing'     smoothing window (in bins, after binning)
%   'makeMovie'     save a movie? (default: false)
%   'alignto'       align spike rate window to saccade onset or ofset
%                   ('fixon' or 'sacon')
%   'usestim'       use pre or postsaccadic stimulus ('pre' or 'post')
% Output:
%   fftrf <struct>      main results
%   plotopts <struct>   additional data for plotting
%
% (c) jly 2020

ip = inputParser();
ip.addParameter('plot', true)
ip.addParameter('win', [-.1 .5])
ip.addParameter('binsize', 1e-3)
ip.addParameter('smoothing', 20)
ip.addParameter('makeMovie', false)
ip.addParameter('alignto', 'fixon', @(x) ismember(x, {'fixon', 'sacon'}))
ip.addParameter('usestim', 'pre', @(x) ismember(x, {'pre', 'post'}))
ip.addParameter('debug', false)
ip.parse(varargin{:})


plotopts = ip.Results;

if isfield(srf, 'fine')
    srf = srf.fine;
end

%% Bin spikes and eye position
sm = ip.Results.smoothing;
win = ip.Results.win; % centered on fixation start
binsize = ip.Results.binsize; % raster resolution
% valid fixations are longer than
fixthresh = 0.2;
    
stimulusSet = 'BackImage';
validTrials = io.getValidTrials(Exp, stimulusSet);

% trial starts and stops
tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

% find all fixation times
fixon = Exp.vpx2ephys(Exp.slist(1:end-1,2));
sacon = Exp.vpx2ephys(Exp.slist(2:end,1));

% exclude fixations outside the ephys recording
bad = (fixon+win(1)) < min(Exp.osp.st) | (fixon+win(2)) > max(Exp.osp.st);
fixon(bad) = [];
sacon(bad) = [];

% which fixations coincided with this stimulus protocol?
[valid, epoch] = getTimeIdx(fixon, tstart, tstop);
fixon = fixon(valid);
sacon = sacon(valid);
fixTrial = epoch(valid);

% sort by fixation duration
fixdur = sacon - fixon;
[~, ind] = sort(fixdur);

% --- eye position
eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1)); % time
remove = find(diff(eyeTime)==0); % bad samples

% filter eye position with 1st order savitzy-golay filter
eyeX = sgolayfilt(Exp.vpx.smo(:,2), 1, 19);
eyeX(isnan(eyeX)) = 0;
eyeY = sgolayfilt(Exp.vpx.smo(:,3), 1, 19);
eyeY(isnan(eyeY)) = 0;

% remove bad samples
eyeTime(remove) = [];
eyeX(remove) = [];
eyeY(remove) = [];

lags = win(1):binsize:win(2);


nlags = numel(lags);
nfix = numel(fixon);
cids = Exp.osp.cids;
NC = numel(cids);

spks = zeros(nfix,NC,nlags);

xpos = zeros(nfix, 2); % pre and post saccade
ypos = zeros(nfix, 2);

[~, ~, id1] = histcounts(fixon, eyeTime);
[~, ~, id2] = histcounts(sacon, eyeTime);

fixindex = zeros(nfix);
for ifix = 1:nfix
    prefix = id1(ifix) + (-100:-50);
    postfix = id1(ifix):(id2(ifix)-20);
    fixindex(ifix) = id1(ifix);
    xpos(ifix,1) = mean(eyeX(prefix));
    xpos(ifix,2) = mean(eyeX(postfix));
    ypos(ifix,1) = mean(eyeY(prefix));
    ypos(ifix,2) = mean(eyeY(postfix));
end

disp('Binning Spikes')
st = Exp.osp.st;
clu = Exp.osp.clu;
keep = ismember(clu, cids);
st = st(keep);
clu = double(clu(keep));

bs = (st==0) + ceil(st/binsize);
spbn = sparse(bs, clu, ones(numel(bs), 1));
spbn = spbn(:,cids);
blags = ceil(lags/binsize);
switch ip.Results.alignto
    case 'fixon'
        balign = ceil(fixon/binsize);
    case 'sacon'
        balign = ceil(sacon/binsize);
end

% Do the binning here
for i = 1:nlags
    spks(:,:,i) = spbn(balign + blags(i),:);
end

disp('Done')

%% For debugging: step through and check individual units
%
% cc = cc+1;
% if cc > NC
%     cc = 1;
% end
% [i,j] = find(squeeze(spks(ind,cc,:)));
% figure(1); clf
% subplot(1,2,1)
% plot.raster(lags(j),i,10)
% axis tight
% title(cc)
% xlabel('Time from Fixation Onset')
%
% subplot(2,2,2)
% plot(srf.fine(cc).sta)
% xlabel('Time lag')
% ylabel('ampltude')
% title('Spatial Mapping')
%
% subplot(2,2,4)
% plot(grf(cc).sta)
% xlabel('Time lag')
% ylabel('ampltude')
% title('Grating Mapping')

%% group units by RF location
% cluster until the each group has a median distance less than 2 d.v.a.
numClusts = 1; % initial number of clusters
condition = false;
thresh = 2;
collapse = true;

rfLocations = srf.rfLocations;
sig = srf.sig;

rfLocations = rfLocations * Exp.S.pixPerDeg;

nonsig = ~(sig);
rfLocations(nonsig,:) = nan;
ix = ~any(isnan(rfLocations),2);

X = rfLocations(ix,:);

Z = linkage(X, 'ward', 'euclidean');

if ip.Results.plot
    figure(1); clf
end

while ~condition

    c = cluster(Z,'Maxclust',numClusts);
    mdist = 0;
    for cc = 1:numClusts
        ii = cc==c;
        meddist = mean(mean( hypot(X(ii,1)-X(ii,1)', X(ii,2)-X(ii,2)'),2)) / Exp.S.pixPerDeg;
        mdist = max(mdist, meddist);
    end
    
    if mdist < thresh
        condition = true;
        fprintf('distance=%02.2f\tFinished with %d clusters\n', mdist, numClusts)
    else
        numClusts = numClusts + 1;
        fprintf('distance=%02.2f\tswitching to %d clusters\n', mdist, numClusts)
    end
        
end


c(nonsig(ix)) = mode(c);
uc = unique(c);
numClusts = numel(uc);
newc = c;
for i = 1:numClusts
    iic = c==uc(i);
    newc(iic) = i;
end
c = newc;    

if collapse
    fprintf('Collapsing clusters with only one unit\n')
    % now collapse all clusters with only one unit in it (they're probably
    % noise)
    uclusts = unique(c);
    goodclusts = uclusts(sum(c==uclusts')>1);
    
    newc = ones(size(c));
    for ic = 1:numel(goodclusts)
        newc(goodclusts(ic)==c) = ic + 1;
    end
else
    newc = c; %#ok<UNRCH>
end

clusts = zeros(size(rfLocations,1), 1);
clusts(ix) = newc;
clusts(clusts==0) = mode(newc);

uclusts = unique(clusts);
numClusts = numel(uclusts);

rois = zeros(numClusts,4);
cmap = lines;

figure(2); clf
for c = 1:numClusts
    ii = clusts==uclusts(c);
    roix = [min(rfLocations(ii,1)) max(rfLocations(ii,1))];
    roiy = [min(rfLocations(ii,2)) max(rfLocations(ii,2))];
    % expand by 20%
    roix = roix + ([-1 1] .* (.2*abs(roix)));
    roiy = roiy + ([-1 1] .* (.2*abs(roiy)));
    
    % ROI must be at least this big
    scaleecc = max(Exp.S.pixPerDeg,1*hypot(mean(rfLocations(ii,1)), mean(rfLocations(ii,2))));
    
    xoff = scaleecc - diff(roix);
    yoff = scaleecc - diff(roiy);
    
    if xoff > 0
        roix = roix + [-1 1].*xoff;
    end
    
    if yoff > 0
        roiy = roiy + [-1 1].*yoff;
    end
    
    rois(c,:) = [roix(1) roiy(1) roix(2) roiy(2)]/Exp.S.pixPerDeg;
    if ip.Results.plot
        plot(rfLocations(ii,1), rfLocations(ii,2), 'o', 'Color', cmap(c,:)); hold on
        plot(roix, roiy([1 1]), 'Color', cmap(c,:))
        plot(roix, roiy([2 2]), 'Color', cmap(c,:))
        plot(roix([1 1]), roiy, 'Color', cmap(c,:))
        plot(roix([2 2]), roiy, 'Color', cmap(c,:))
    end
end

if ip.Results.plot
    xlabel('Position (pixels)')
    ylabel('Position (pixels)')
    title('RF clusters')
end

rfLocations = rfLocations / Exp.S.pixPerDeg;


if ip.Results.debug
    keyboard
end
   
%% parameters of analysis

ppd = Exp.S.pixPerDeg;
ctr = Exp.S.centerPix;


% initialize output
stmp = [];
stmp.lags = [];
stmp.rfLocation = [];
stmp.rateHi = [];
stmp.rateLow = [];
stmp.stdHi = [];
stmp.stdLow = [];
stmp.nHi = [];
stmp.nLow = [];
stmp.rf.kx = [];
stmp.rf.ky = [];
stmp.rf.Ifit = [];
stmp.xproj.bins = [];
stmp.xproj.cnt = [];
stmp.cid = [];

fftrf = repmat(stmp, NC, 1);

switch ip.Results.usestim
    case 'pre'
        usestim = 1;
    case 'post'
        usestim = 2;
end

% setup movie
exname = strrep(Exp.FileTag, '.mat', '');
if nargout > 1
    plotopts.clusts = clusts;
    clustMeta = repmat(struct('rfCenter', [], 'rect', [], 'fixIms', [], 'fftIms', [],...
        'fixMeta', []), max(clusts), 1);
end

for clustGroup = unique(clusts(:)')
    
    fig = figure(2);
    fig.Position = [100 100 800 250]; clf
    fig.Color = 'w';
    
    fname = sprintf('Figures/fftfixclip_%s_%d', exname, clustGroup);
    if ip.Results.makeMovie
        vidObj = VideoWriter(fname, 'MPEG-4');
        vidObj.FrameRate = 3;
        vidObj.Quality = 100;
        open(vidObj);
    end
    
    iic = clusts(:)==clustGroup & srf.sig(:);
    if sum(iic) == 1
        rfCenter = rfLocations(iic,:);
    else
        rfCenter = nanmedian(rfLocations(iic,:));
    end
    
    rfwidth = hypot(rfCenter(1), rfCenter(2));
    rfwidth = max(rfwidth, 1);
    rfwidth = min(rfwidth, 3);
    rect = [-1 -1 1 1]*ceil(ppd*rfwidth); % window centered on RF
    dims = [rect(4)-rect(2) rect(3)-rect(1)];
    
    
    hwin = hanning(dims(1))*hanning(dims(2))';
    
    nxfft = 2^nextpow2(dims(2));
    nyfft = 2^nextpow2(dims(1));
    
%     nxfft = dims(2);
%     nyfft = dims(1);
    
    xax = -nxfft/2:nxfft/2;
    xax = xax / nxfft * ppd;
    
    fixIms = zeros(dims(1), dims(2), nfix, 2);
    fftIms = zeros(nyfft, nxfft, nfix, 2);
    fixMeta = zeros(nfix, 4); % trial, x, y
    
    %% do fixation clipping and compute fft
    nTrials = numel(validTrials);
    if ip.Results.makeMovie
        nTrials = 4;
    end
    
    for iTrial = 1:nTrials
        
        fprintf('%d/%d\t', iTrial, numel(validTrials))
        
        thisTrial = validTrials(iTrial);
        
        % load image
        try
            Im = imread(fullfile(fileparts(which('marmoV5')), Exp.D{thisTrial}.PR.imagefile));
        catch
            try
                Im = imread(fullfile(fileparts(which('marmoV5')), strrep(Exp.D{thisTrial}.PR.imageFile, '\', filesep)));
            catch
                fprintf(1, 'regenerateStimulus: failed to load image\n')
                continue
            end
        end
        
        % zero mean
        Im = mean(Im,3)-127;
        Im = imresize(Im, fliplr(Exp.S.screenRect(3:4)));
        
        % loop over fixations
        fixtrial = find(validTrials(fixTrial) == validTrials(iTrial));
        nft = numel(fixtrial);
        if ip.Results.plot
            figure(1); clf
            imagesc(Im); hold on
            colormap gray
            plot(xpos(fixtrial,:)*ppd + ctr(1), -ypos(fixtrial,:)*ppd + ctr(2), 'ro')
            drawnow
        end
        
        fprintf('%d fixations...\n', nft)
        % loop over fixations
        for ifix = 1:nft
            
            thisfix = fixtrial(ifix);
            eyeX = xpos(thisfix, usestim)*ppd + ctr(1) + rfCenter(1)*ppd;
            eyeY = -ypos(thisfix, usestim)*ppd + ctr(2) - rfCenter(2)*ppd;
            
            % center on eye position
            tmprect = rect + [eyeX eyeY eyeX eyeY];
            
            imrect = [tmprect(1:2) (tmprect(3)-tmprect(1))-1 (tmprect(4)-tmprect(2))-1];
            I = imcrop(Im, imrect); % requires the imaging processing toolbox
            
            if ip.Results.makeMovie
                figure(fig)
                subplot(131, 'align')
                imagesc(Im); hold on; colormap gray
                axis off
                plot(eyeX, eyeY, '.c', 'MarkerSize', 10)
                plot([imrect(1) imrect(1) + imrect(3)], imrect([2 2]), 'r', 'Linewidth', 2)
                plot([imrect(1) imrect(1) + imrect(3)], imrect(2)+imrect([4 4]), 'r', 'Linewidth', 2)
                plot(imrect([1 1]),[imrect(2), imrect(2) + imrect(4)], 'r', 'Linewidth', 2)
                plot(imrect(1)+imrect([3 3]), [imrect(2), imrect(2) + imrect(4)], 'r', 'Linewidth', 2)
            end
            
            if ~all(size(I)==dims(1))
                continue
            end
            
            Iwin = (I - mean(I(:))).*hwin;
            % window
%             Iwin = I.*hwin;
            
            if ip.Results.makeMovie
                subplot(132, 'align')
                imagesc(Iwin, [-1 1]*max(abs(Iwin(:))))
                axis off
            end
            
%             fIm = fftshift(fft2(Iwin));
            fIm = fftshift(fft2(Iwin, nxfft, nyfft));
            
            
            if ip.Results.makeMovie
                subplot(133, 'align')
                imagesc(xax, xax, abs(fIm))
                xlabel('SF_x')
                ylabel('SF_y')
                drawnow
                
                currFrame = getframe(gcf);
                writeVideo(vidObj, currFrame)
            end
            
            fixIms(:,:,thisfix,2) = Iwin;
            fftIms(:,:,thisfix,2) = fIm;
            fixMeta(thisfix,1) = thisTrial;
            fixMeta(thisfix,2:3) = [eyeX eyeY];
            fixMeta(thisfix,4) = fixindex(thisfix);
            
        end
        
        
    end
    
    if nargout > 1
        clustMeta(clustGroup).rfCenter = rfCenter;
        clustMeta(clustGroup).rect = rect;
        clustMeta(clustGroup).fixIms = fixIms;
        clustMeta(clustGroup).fftIms = fftIms;
        clustMeta(clustGroup).fixMeta = fixMeta;
    end
    
    if ip.Results.makeMovie
        close(vidObj)
    end
    
    
    %% Loop over units in group
    
    % integration windows for spikes
    Bearly = exp(- ((lags*1e3 - 80).^2/2/20^2));
    Bearly = Bearly ./ sum(Bearly);
    Blate = 1 ./ (1 + exp(- (lags*1e3 - 100) / 20));
    Blate(lags>.3) = 0;
    Blate = Blate ./ sum(Blate);
    
    
        
        
    clist = find(clusts==clustGroup);
    
    % loop over cells in this cluster
    for cc = clist(:)'
        
        [kx, ky] = meshgrid(xax(2:end), xax(2:end));
        [ori, cpd] = cart2pol(ky, kx);
        ori = wrapToPi(ori);
        
        params = grf.rffit(cc).pHat;
        if isfield(grf.rffit(cc), 'fitfun')
            fitfun = grf.rffit(cc).fitfun;
        else
            fitfun = 'parametric_rf';
        end
        
        switch fitfun
            case 'polarRF'
                Ifit = prf.polarRF([ori(:)/pi*180 cpd(:)], params(1), params(2), params(3), params(4), params(5), params(6));
            case 'parametric_rf'
                lg = strcmp(grf.sftuning, 'loggauss');
                if ~isempty(params)
                    Ifit = prf.parametric_rf(params, [ori(:), cpd(:)], lg);
                else
                    Ifit = randn(size(ori));
                end
        end
        
        
        oriPref = grf.rffit(cc).oriPref;
        if oriPref < 0
            oriPref = 180 + oriPref;
        end
        
        rf = reshape(Ifit, size(ori));
        if ip.Results.plot
            figure(1); clf
            set(gcf, 'Color', 'w')
            subplot(2,2,1)
            imagesc(xax, xax, rf)
            title(sprintf('RF: %02.2f', oriPref))
            xlabel('sf')
            ylabel('sf')
        end
        
        freshape = abs(reshape(fftIms(:,:,:,2), [nxfft*nyfft nfix]));
        
        fproj = freshape'*rf(:);
        
        good_trials= find(sum(freshape)~=0 & fixdur' > fixthresh);
        if ip.Results.plot
            subplot(2,2,2)
            
        end
        
        % get histogram of generator signal
        [binCnt, binEdges] = histcounts(fproj(good_trials), 25);
        binCenters = binEdges(1:end-1)+mean(diff(binEdges))/2;
        
        % percentile levels for generator signal
        pref_null_levels = prctile(fproj(good_trials), [10 90]);
        
        lowix = good_trials(fproj(good_trials) < pref_null_levels(1));
        hiix = good_trials(fproj(good_trials) > pref_null_levels(2));
        
        if ip.Results.plot
            hold on
            plot(pref_null_levels(1)*[1 1], ylim, 'r', 'Linewidth', 2)
            plot(pref_null_levels(2)*[1 1], ylim, 'b', 'Linewidth', 2)
            
            xlabel('Generator Signal')
            ylabel('Fixation Count')
            
            % psth
            subplot(2,2,3)
        end
        
        
%         assert(sum(binCenters > pref_null_levels(2)) > 1)
        
        spksfilt = filter(ones(sm,1), 1, squeeze(spks(:,cc,:))')';
        
        psthhi = mean(spksfilt(hiix,:))/binsize/sm;
        psthlow = mean(spksfilt(lowix, :))/binsize/sm;
        
        pstvhi = std(spksfilt(hiix,:))/binsize/sm;
        pstvlow = std(spksfilt(lowix, :))/binsize/sm;
        
        
           
       
%         figure(2); clf
%         plot(lags, Bearly); hold on
%         plot(lags, Blate)
        
        % get pearson correlation between spikes and fftIm -- this is a
        % sort of reverse correlation
        y = spksfilt(good_trials,:);
        X = abs(reshape(fftIms(:,:,:,2), [nxfft*nyfft nfix]));
        X = X(:,good_trials);
        
        [rho, pval] = corr(fproj(good_trials,:), spksfilt(good_trials,:));
        
        nHi = numel(hiix);
        nLow = numel(lowix);
        
        if ip.Results.plot
            plot.errorbarFill(lags, psthhi, pstvhi/sqrt(numel(hiix)), 'b', 'FaceColor', 'b'); hold on
            plot.errorbarFill(lags, psthlow, pstvlow/sqrt(numel(lowix)), 'r', 'FaceColor', 'r');
            axis tight
            xlim([-.05 .25])
            title(cc)
            xlabel('Time from fix on (s)')
            ylabel('Firing Rate')
            
            subplot(2,2,4)
            plot(lags, rho, 'k'); hold on
            xlim([-.05 .25])
            xlabel('Time from fix on (s)')
            ylabel('Ratio (pref / null)')
            
        end

        steps = [.5 .15];
        nsteps = numel(steps);
        frf = zeros([size(fIm) nsteps]);
        
        ys = y*Bearly';
        ysE = ys - mean(ys);
        frf(:,:,1) = imgaussfilt(reshape(X*ysE./sum(X,2), size(fIm)));

        ys = sum(y*Blate',2);
        ysL = ys - mean(ys);
        frf(:,:,2) = imgaussfilt(reshape(X*ysL./sum(X,2), size(fIm)), 1);
        
        % filter the stimulus in the orientation band but at different SFs
        sfs = [1 2 4 8 10 16];
        
        switch fitfun
            case 'polarRF'
                sfbs = params(5)*[5 1 .5 .1 .5 1];
                Ifits = nan(numel(ori), numel(sfs));
                for i = 1:numel(sfs)
                    Ifits(:,i) = prf.polarRF([ori(:)/pi*180 cpd(:)], params(1), params(2), sfs(i), params(4), sfbs(i), params(6));
                end
            case 'parametric_rf'
                
                
                lg = strcmp(grf.sftuning, 'loggauss');
                Ifits = nan(numel(ori), numel(sfs));
                sfbs = nan*[5 1 .5 .1 .5 1];
                if ~isempty(params)
                    sfbs = params(4)*[5 1 .5 .1 .5 1];
                    for i = 1:numel(sfs)
                        params(3) = sfs(i);
                        params(4) = sfbs(i);
                        Ifits(:,i) = reshape(prf.parametric_rf(params, [ori(:), cpd(:)], lg), [], 1);
                    end
                end
        end

%         figure(2); clf
        rhos = zeros(numel(lags), numel(sfs));
        fprojs = freshape'*Ifits;
        
        vinds = double(lags <= fixdur(good_trials));
        y = spksfilt(good_trials,:).*vinds;
        for i = 1:numel(sfs)
%             subplot(1,numel(sfs), i)
%             imagesc(reshape(Ifits(:,i), size(ori)))
            
            [rhos(:,i), ~] = corr(fprojs(good_trials,i), y);
        end
        
        %
%         figure(3); clf
%         cmap = jet(numel(sfs));
%         for i = 1:numel(sfs)
%             plot(lags, rhos(:,i), 'Color', cmap(i,:)); hold on
%         end
%         xlabel('Time from fixation')
%         ylabel('Pearson Correlation Coefficient')
%         legstr = arrayfun(@(x) sprintf('sf: %d', x), sfs, 'uni', 0);
%         legend(legstr{:})
        
        
%         figure(4); clf
%         plot(sfs, rhos'*Bearly', '-o'); hold on
%         plot(sfs, rhos'*Blate', '-o')
%         xlabel('Spatial Frequency')
%         ylabel('Rho')
%         legend({'Early spikes', 'Late Spikes'})
        
        mxspks = max(psthhi)*1.1;
        
        nsf = numel(sfs);
        sfpsthhi = zeros(nlags, nsf);
        sfpsthlow = zeros(nlags, nsf);
        sfpstvhi = zeros(nlags, nsf);
        sfpstvlow = zeros(nlags, nsf);
        
        if ip.Results.plot
            figure(5); clf
        end
        
        % loop over spatial frequencies, re compute levels and null pref
        % plot
        for i = 1:nsf
            fproj = fprojs(:,i);
            levels = prctile(fproj(good_trials), [10 90]);
        
            lowix = good_trials(fproj(good_trials) < levels(1));
            hiix = good_trials(fproj(good_trials) > levels(2));
        
            sfpsthhi(:,i) = mean(spksfilt(hiix,:))/binsize/sm;
            sfpsthlow(:,i) = mean(spksfilt(lowix, :))/binsize/sm;
        
            sfpstvhi(:,i) = std(spksfilt(hiix,:))/binsize/sm;
            sfpstvlow(:,i) = std(spksfilt(lowix, :))/binsize/sm;
            
            if ip.Results.plot
                try
                subplot(1, numel(sfs), i)
                plot.errorbarFill(lags, sfpsthhi(:,i), sfpstvhi(:,i)/sqrt(numel(hiix)), 'b', 'FaceColor', 'b'); hold on
                plot.errorbarFill(lags, sfpsthlow(:,i), sfpstvlow(:,i)/sqrt(numel(lowix)), 'r', 'FaceColor', 'r');
                ylim([0 mxspks(1)])
                xlim([-0.05 .3])
                title(sfs(i))
                end
            end
        end
        
        if ip.Results.plot
            figure(10); clf
            clim = [min(frf(:)) max(frf(:))];
            for i = 1:nsteps
                subplot(1,nsteps+1, i)
                imagesc(frf(:,:,i), clim)
            end
            
            subplot(1,nsteps+1,nsteps+1)
            imagesc(xax, xax, rf)
            colormap(viridis)
            drawnow
        end
        
        fftrf(cc).sftuning = struct();
        fftrf(cc).sftuning.B = [Bearly(:) Blate(:)];
        fftrf(cc).sftuning.sfs = sfs;
        fftrf(cc).sftuning.sfbw = sfbs;
        fftrf(cc).sftuning.rhos = rhos;
        fftrf(cc).sftuning.sfpsthhi = sfpsthhi;
        fftrf(cc).sftuning.sfpsthlow = sfpsthlow;
        fftrf(cc).sftuning.sfpstvhi = sfpstvhi;
        fftrf(cc).sftuning.sfpstvlow = sfpstvlow;
        
        fftrf(cc).frfsteps = steps;
        fftrf(cc).frf = frf;
        % save 
        fftrf(cc).lags = lags;
        fftrf(cc).rfLocation = rfCenter;
        fftrf(cc).rateHi = psthhi;
        fftrf(cc).rateLow = psthlow;
        fftrf(cc).stdHi = pstvhi;
        fftrf(cc).stdLow = pstvlow;
        fftrf(cc).nHi = nHi;
        fftrf(cc).nLow = nLow;
        fftrf(cc).rf.kx = xax;
        fftrf(cc).rf.ky = xax;
        fftrf(cc).rf.Ifit = reshape(Ifit, size(kx));
        fftrf(cc).xproj.bins = binCenters;
        fftrf(cc).xproj.cnt = binCnt;
        fftrf(cc).xproj.levels = pref_null_levels;
        fftrf(cc).cid = grf.rffit(cc).cid;
        fftrf(cc).rect = rect;
        fftrf(cc).corrrho = rho;
        fftrf(cc).corrp = pval;
        
        [i,j] = find(squeeze(spks(ind,cc,:)));
        
        if ip.Results.plot
            figure(2); clf
            subplot(1,2,1)
            plot.raster(j,i,10);
            axis tight
            title(cc)
            
            subplot(2,2,2)
            imagesc(srf.spatrf(:,:,cc))
            title('spatial sta')
            
            subplot(2,2,4)
            if isfield(grf.rffit(cc), 'srf')
                imagesc(grf.rffit(cc).srf)
            else
                plot.polar_contourf(grf.xax, grf.yax, grf.srf(:,:,cc))
            end
                
            title('hartley sta')
            xlabel('time lags')
            
            drawnow
            if ip.Results.debug
                keyboard
            end
            
        end
    end
end


%%

if nargout > 1
    plotopts.clustMeta = clustMeta;
    plotopts.spks = spks;
    plotopts.fixdur = fixdur;
end

%%

