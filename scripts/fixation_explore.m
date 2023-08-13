%% Overview
% This script demonstrates extracting a 

figure(4); clf


dt = median(diff(Exp.vpx.raw0(:,1)));
win = 0.041;
win = ceil(win/dt);


plot(Exp.vpx.raw0(:,1), Exp.vpx.raw0(:,2))
hold on
plot(Exp.vpx.raw0(:,1), sgolayfilt(Exp.vpx.raw0(:,2), 3, win))
hold on
plot(Exp.vpx.raw0(:,1), sgolayfilt(Exp.vpx.raw0(:,2), 1, 19))

% plot(Exp.vpx.raw(:,1), Exp.vpx.raw(:,2))

%%


% plot(Exp.vpx.smo(:,2))
hold on

Bx = robustfit(Exp.vpx.raw(:,2), Exp.vpx.smo(:,2));
By = robustfit(Exp.vpx.raw(:,3), Exp.vpx.smo(:,3));

eyeX = Exp.vpx.raw(:,2)*Bx(2) + Bx(1);
eyeY = Exp.vpx.raw(:,3)*By(2) + By(1);

eyeX = sgolayfilt(eyeX, 3, 41);

plot(eyeX, '-r')
% 

%%


stimulusSet = 'BackImage';
validTrials = io.getValidTrials(Exp, stimulusSet);

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
M = 300;
N = 300;

[xi, yi] = meshgrid(linspace(-10, 10, 101));
theta = pi/4;
gi = sin(theta)*yi + cos(theta)*xi;
I = sin(5*gi);

figure(1); clf
subplot(2,2,1)
imagesc(I)
subplot(2,2,2)
imagesc((fftshift(abs(fft2(I-mean(I(:)), M, N)))))

win = hamming(101)*hamming(101)';
win = sqrt(win);
subplot(2,2,3)
Itmp = (I-mean(I(:))) .* win ;
imagesc(Itmp)

subplot(2,2,4)
imagesc((fftshift(abs(fft2( Itmp , M, N)))))

%%

iTrial = 0;
%%

%% reconstruct stimulus
rect = [-20 -10 50 60];
% rect = [0 0 1280 720]
thisTrial = validTrials(10);
validTrials = io.getValidTrials(Exp, 'BackImage');
[Stim2, frameInfo2] = regenerateStimulus(Exp, thisTrial, rect, 'spatialBinSize', 1, ...
    'EyePos', [eyeX eyeY], ...
    'debug', false, ...
    'Latency', 0, 'includeProbe', true, 'GazeContingent', true, 'ExclusionRadius', inf,...
    'frameIndex', [], 'usePTBdraw', false);


% %%
% [Stim2, frameInfo2] = regenerateStimulus(Exp, validTrials(1), rect, 'spatialBinSize', 1, ...
%     'EyePos', [eyeX eyeY], ...
%     'Latency', 0, 'includeProbe', true, 'GazeContingent', true, 'ExclusionRadius', inf,...
%     'frameIndex', [], 'usePTBdraw', true);

%%

figure(1); clf

for i = 1:size(Stim,3)

    imagesc(Stim(:,:,i))
    drawnow
end

%%

validTrials = io.getValidTrials(Exp, 'BackImage');
thisTrial = validTrials(10);
%%
figure(1); clf; 
plot(Exp.D{thisTrial}.eyeData(:,1), Exp.D{thisTrial}.eyeData(:,5))

hold on
plot(Exp.D{thisTrial}.eyeData(:,1), Exp.D{thisTrial}.eyeData(:,2)-130)

plot(Exp.D{thisTrial}.PR.NoiseHistory(:,1), Exp.D{thisTrial}.PR.NoiseHistory(:,4))
plot(frameInfo.frameTimesPtb, squeeze(mean(mean(Stim.^2, 2),1))/100)
plot(frameInfo.frameTimesPtb, frameInfo.eyeAtFrame(:,2)-600)

figure(2); clf; 
plot(Exp.D{thisTrial}.eyeData(:,1), Exp.D{thisTrial}.eyeData(:,5))

hold on
plot(Exp.D{thisTrial}.eyeData(:,1), Exp.D{thisTrial}.eyeData(:,2)-130)

plot(Exp.D{thisTrial}.PR.NoiseHistory(:,1), Exp.D{thisTrial}.PR.NoiseHistory(:,4))

%%
figure(1); clf
plot(frameInfo2.frameTimesPtb, squeeze(mean(mean(Stim2.^2, 2),1))/100); hold on
plot(frameInfo2.frameTimesPtb, frameInfo2.eyeAtFrame(:,2)-600)

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
figure(1); clf
subplot(1,2,1)
imagesc(Stim(:,:,1))
subplot(1,2,2)
imagesc(Stim2(:,:,4))

%%
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
    pause(.1)

% 
% 
%     win = [0 0];
%     
%     for ifix = 1:nfix
%         %     if ifix > nfix
%         %         ifix = 0;
%         %     end
% 
%         thisfix = fixix(ifix);
% 
%         ii = eyeTime > fixon(thisfix)+win(1) & eyeTime < sacon(thisfix)+win(2);
% 
%         et = eyeTime(ii);
%         fixX = eyeX(ii)*ppd + ctr(1);
%         fixY = -eyeY(ii)*ppd + ctr(2);
% 
% 
%         eyeFiltX = filter(bcEye, 1, eyeX(ii));
%         eyeFiltY = filter(bcEye, 1, eyeY(ii));
% 
%         eyeFiltX = eyeFiltX(1:nbc:end);
%         eyeFiltY = eyeFiltY(1:nbc:end);
% 
%         binfun = @(t) (t==0) + ceil(t/binsize);
% 
%         iix = st > fixon(thisfix)+win(1) & st < sacon(thisfix)+win(2);
%         st_ = st(iix);
%         clu_ = clu(iix);
%         
%         n = size(eyeFiltX,1);
%         spk = sparse(binfun(st_ - fixon(thisfix)), double(clu_), ones(numel(clu_), 1), n+1, MC);
%         spk = full(spk(1:n,:));
%     
% 
%         gazeFixX(1:n,numFixCumulative(iTrial)+ifix) = eyeFiltX;
%         gazeFixY(1:n, numFixCumulative(iTrial)+ifix) = eyeFiltY;
%         spikesFix(1:n,:,numFixCumulative(iTrial)+ifix) = spk(:,cids);
%     end
% end

%%
dx = gazeFixX(1:numBins,:) - gazeFixX(1,:);

dx = dx(:,sum(abs(diff(dx)) > 1) == 0);
dx = dx(5:end,:); dx = dx - dx(1,:);

dx(isnan(dx)) = 0;
plot(dx)

[Pxx, F] = pwelch(dx, [], [], [], 1/binsize);

figure(4); clf
plot(F, mean(Pxx,2))

figure(3); clf
imagesc(gazeFixX(1:numBins,:) - gazeFixX(1,:))

%%
cc = 1;
[i,j] = find(squeeze(spikesFix(1:numBins,cc,:)));
plot.raster(i,j)


%%
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
plot(binedges(1:end-1)-fixon(thisfix), imgaussfilt(cnt, 2), 'k')

subplot(2,1,2)

st_ = st(iix);
clu_ = clu(iix);

for i = 1:nbins
    plot.raster(st_(binid==i)-fixon(thisfix), clu_(binid==i), 5, 'Linewidth', 2, 'Color', cmap(i,:)); hold on
end
xlim(xd)


%%
binsize = 5e-3;
binfun = @(t) (t==0) + ceil(t/binsize);
spk = sparse(binfun(st_ - fixon(thisfix)), double(clu_), ones(numel(clu_), 1));

spk = full(spk');

[Pxx, F] = pwelch(spk, [], [], [], 1/binsize);

figure(1); clf
plot(F, mean(Pxx,2))
%%
figure(3); clf
plot(spk(:,6))

%%
clf

Iwin = log10(I+127).*hwin;
% Iwin = (I - mean(I(:))).*hwin;

imagesc(Iwin, [-1 1]*max(abs(Iwin(:))))


axis off

%%
fIm = fftshift(fft2(Iwin, nxfft, nyfft));
            
            
            



%%
[valid, epoch] = getTimeIdx(fixon, tstart, tstop);
fixon = fixon(valid);
sacon = sacon(valid);
fixTrial = epoch(valid);

fixdur = sacon - fixon;

% --- eye position
eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1)); % time
remove = find(diff(eyeTime)==0); % bad samples


% filter eye position with 1st order savitzy-golay filter
eyeX = sgolayfilt(Exp.vpx.smo(:,2), 1, 31);
eyeX(isnan(eyeX)) = 0;
eyeY = sgolayfilt(Exp.vpx.smo(:,3), 1, 31);
eyeY(isnan(eyeY)) = 0;

% remove bad samples
eyeTime(remove) = [];
eyeX(remove) = [];
eyeY(remove) = [];

binSize = median(diff(eyeTime));

[~, ~, id1] = histcounts(fixon, eyeTime);
[~, ~, id2] = histcounts(sacon, eyeTime);


nFix = numel(fixon);
drifts = cell(nFix,1);
for ifix = 1:nFix
    
    
    
    fixix = id1(ifix):id2(ifix);
    fixix = fixix(30:end);
    
    if numel(fixix) < 200
        continue
    end
    
    ctrx = eyeX(fixix(1));
    ctry = eyeY(fixix(1));

    
    fixX = (eyeX(fixix)-ctrx)*60;
    fixY = (eyeY(fixix)-ctry)*60;
    spd = hypot(diff(fixX), diff(fixY));
    nxsac = find(spd > 1, 1);
    if isempty(nxsac)
        nxsac = numel(spd);
    end
    nxsac = nxsac - 15;
    
    fixX = fixX(1:nxsac);
    fixY = fixY(1:nxsac);
    
    drift = hypot(fixX, fixY);
    drifts{ifix} = drift;
end