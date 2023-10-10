%% Overview

% Note: requires and processed Exp struct to be in the workspace


% this is how you get which trials are which
stimulusSet = 'BackImage'; % Gabor, Grating, FixRsvpStim
validTrials = io.getValidTrials(Exp, stimulusSet);

% this is when the trials started and stopped in the same clock as the
% ephys
tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

% fixation times
fixon = Exp.vpx2ephys(Exp.slist(1:end-1,2));
sacon = Exp.vpx2ephys(Exp.slist(2:end,1));


eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
eyeX = Exp.vpx.smo(:,2);
eyeY = Exp.vpx.smo(:,3);

cids = unique(Exp.osp.clu);
NC = numel(cids);
MC = max(cids);
eyesmoothing = 41;
eyeX = sgolayfilt(eyeX, 3, eyesmoothing);
eyeY = sgolayfilt(eyeY, 3, eyesmoothing);




%% Prepare for storing 
ppd = Exp.S.pixPerDeg;
ctr = Exp.S.centerPix;
rect = [-1 -1 1 1]*ceil(ppd*1); % window centered on RF
dims = [rect(4)-rect(2) rect(3)-rect(1)];
hwin = hanning(dims(1))*hanning(dims(2))';
maxFix = .5;
binsize = 5e-3;

numFixPerTrial = arrayfun(@(x,y) sum(fixon > x & fixon < y), tstart, tstop);
numFixTotal = sum(numFixPerTrial);
numFixCumulative = [0; cumsum(numFixPerTrial)];

numBins = ceil(maxFix/binsize);

gazeFixX = nan(numBins, numFixTotal);
gazeFixY = nan(numBins, numFixTotal);
spikesFix = nan(numBins, NC, numFixTotal);

dtEye = median(diff(eyeTime));
nbc = ceil(binsize/dtEye);
bcEye = ones(nbc,1)/nbc;
% numFreq
% stimFix = nan(numFreq, numFixTotal);





%%

iTrial = 0;
%%

%% reconstruct stimulus
% pick a trial and replay the stimulus
rect = [-20 -10 50 60];
% rect = [0 0 1280 720]
validTrials = io.getValidTrials(Exp, 'Gabor');
thisTrial = validTrials(10);

[Stim, frameInfo] = regenerateStimulus(Exp, thisTrial, rect, 'spatialBinSize', 1, ...
    'EyePos', [eyeX eyeY], ...
    'debug', false, ...
    'Latency', 0, 'includeProbe', true, 'GazeContingent', true, 'ExclusionRadius', inf,...
    'frameIndex', [], 'usePTBdraw', false);

disp("Done")

%% play a movie of that stimulus

figure(1); clf

for i = 1:size(Stim,3)

    imagesc(Stim(:,:,i))
    drawnow
end


%%
et = frameInfo.eyeAtFrame(:,1);

figure(1); clf
plot(frameInfo.eyeAtFrame(:,1), frameInfo.eyeAtFrame(:,2), '-');
hold on
yd = ylim ;
for i = 1:size(frameInfo.blocks,1)
    block = frameInfo.blocks(i,:);
    fill(et(block([1 1 2 2])),[ylim fliplr(ylim)], 'r', 'FaceAlpha', .25)
end


%%
%% pick an image trial

validTrials = io.getValidTrials(Exp, 'BackImage');
iTrial = 1;
%%
iTrial = iTrial + 1;
% for iTrial = 1:numel(validTrials)
        
    fprintf('%d/%d\t', iTrial, numel(validTrials))
            
    thisTrial = validTrials(iTrial);
            
    % load image
    try
        Im = imread(fullfile(fileparts(which('marmoV5')), Exp.D{thisTrial}.PR.imagefile));
    catch
        try
            Im = imread(fullfile(fileparts(which('marmoV5')), strrep(Exp.D{thisTrial}.PR.imageFile, '\', filesep)));
        catch
            error('regenerateStimulus: failed to load image\n')
        end
    end
            
    % zero mean
    Im = mean(Im,3)-127;
    Im = imresize(Im, fliplr(Exp.S.screenRect(3:4)));
            
    figure(1); clf
    imagesc(Im); hold on
    colormap gray
    
    eyeix = eyeTime > tstart(iTrial) & eyeTime < tstop(iTrial);
    
    figure(2); clf
    ax = axes();
    plot(eyeTime(eyeix), eyeX(eyeix)); hold on
    plot(eyeTime(eyeix), eyeY(eyeix));
            
    % fprintf('%d fixations...\n', nft)
    % loop over fixations
    
    fixix = find(fixon > tstart(iTrial) & fixon < tstop(iTrial));
    nfix = numel(fixix);
    
    for ifix = 1:nfix
        x = [fixon(fixix(ifix))*[1 1] sacon(fixix(ifix))*[1 1]];
        y = [ylim fliplr(ylim)];
        fill(x,y,'r', 'FaceAlpha', .1, 'EdgeColor', 'none')
    %     fill(*[1 1], ylim, 'r')
    end
    
    
    % ax2 = axes('Position', ax.Position);
    
    ifix = 0;
    
    spkix = Exp.osp.st > tstart(iTrial) & Exp.osp.st < tstop(iTrial);
    st = Exp.osp.st(spkix);
    clu = Exp.osp.clu(spkix);
    plot.raster(st, clu+10, 2, 'k', 'Linewidth', 2)
    axis off
%     pause(.1)

ifix = 0;

%%    


win = [0 -0.05];
ifix = ifix + 1;
%     
%     for ifix = 1:nfix
if ifix > nfix
    ifix = 0;
end
% 
thisfix = fixix(ifix);

ii = eyeTime > fixon(thisfix)+win(1) & eyeTime < sacon(thisfix)+win(2);

et = eyeTime(ii);
fixX = eyeX(ii)*ppd + ctr(1);
fixY = -eyeY(ii)*ppd + ctr(2);


eyeFiltX = filter(bcEye, 1, eyeX(ii));
eyeFiltY = filter(bcEye, 1, eyeY(ii));

eyeFiltX = eyeFiltX(1:nbc:end);
eyeFiltY = eyeFiltY(1:nbc:end);

binfun = @(t) (t==0) + ceil(t/binsize);

iix = st > fixon(thisfix)+win(1) & st < sacon(thisfix)+win(2);
st_ = st(iix);
clu_ = clu(iix);


stbin = binfun(st_ - fixon(thisfix));
n = max(size(eyeFiltX,1), max(stbin));
spk = sparse(stbin, double(clu_), ones(numel(clu_), 1), n+1, MC);
spk = full(spk(1:n,:));
    

% center on eye position
i = ceil(numel(et)/2);
tmprect = rect + [fixX(i) fixY(i) fixX(i) fixY(i)];

imrect = [tmprect(1:2) (tmprect(3)-tmprect(1))-1 (tmprect(4)-tmprect(2))-1];
I = imcrop(Im, imrect); % requires the imaging processing toolbox

figure(1); clf

% plot(et, eyeX(ii))


nbins = 500;
binedges = linspace(fixon(thisfix)+win(1), sacon(thisfix)+win(2), nbins);
cmap = parula(nbins);

[~, ~, binid] = histcounts(et, binedges);

ax = subplot(2,2,1:2);
xx = (fixX-ctr(1))/ppd*60;
x0 = mean(xx);
plot(et-fixon(thisfix), xx-x0, 'k'); hold on
for i = 1:nbins
    plot(et(binid==i)-fixon(thisfix), (fixX(binid==i)-ctr(1))/ppd*60-x0, '.', 'MarkerSize', 10, 'Color', cmap(i,:));
end
xd = xlim;

figure(2); clf
imagesc(Im); colormap gray
% imagesc(rect(1):rect(3), rect(2):rect(4), I)
hold on
for i = 1:nbins
    plot(fixX(binid==i), fixY(binid==i), '.', 'MarkerSize', 10, 'Color', cmap(i,:));
end


% plot(fixX, fixY, 'r', 'Linewidth', 2)
xlim(tmprect([1 3]))
ylim(tmprect([2 4]))


iix = st > fixon(thisfix)+win(1) & st < sacon(thisfix)+win(2);
[cnt, ~, binid] = histcounts(st(iix), binedges);


figure(1)
subplot(2,1,1)
plot(binedges(1:end-1)-fixon(thisfix), cnt, 'k')

subplot(2,1,2)

st_ = st(iix);
clu_ = clu(iix);

for i = 1:nbins
    plot.raster(st_(binid==i)-fixon(thisfix), clu_(binid==i), 5, 'Linewidth', 2, 'Color', cmap(i,:)); hold on
end
xlim(xd)

%%
tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));
stimLatency = 8e-3;

cids = unique(Exp.osp.clu);
ev = Exp.ptb2Ephys(cellfun(@(x) x.PR.startTime, Exp.D(validTrials))) + stimLatency;
evsacon = Exp.vpx2ephys(Exp.slist(:,1));
evsacon = evsacon(getTimeIdx(evsacon, tstart, tstop));
evsacoff = Exp.vpx2ephys(Exp.slist(:,2));
evsacoff = evsacoff(getTimeIdx(evsacoff, tstart, tstop));

win = [-.5 1];
bs = 10e-3;
NC = numel(cids);

cc = cc + 1;
if cc > NC
    cc = 0;
end


sptimes = Exp.osp.st(Exp.osp.clu==cids(cc));



[m,s,bc,v, tspcnt] = eventPsth(sptimes, ev, win, bs);
[msacon,ssacon,~,vsacon, tspcntsacon] = eventPsth(sptimes, evsacon, win, bs);
[msacoff,ssacoff,~,vsacoff, tspcntsacoff] = eventPsth(sptimes, evsacoff, win, bs);

figure(1); clf
set(gcf, 'Color', 'w')
plot(bc, m, 'Linewidth', 2); hold on
plot(bc, msacon, 'Linewidth', 2)
plot(bc, msacoff, 'Linewidth', 2)
legend({'Image Onset', 'Saccade Onset', 'Saccade Offset'})
title(cc)