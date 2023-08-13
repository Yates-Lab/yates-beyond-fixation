%% Overview


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
% eyesmoothing = 41;
% eyeX = sgolayfilt(eyeX, 3, eyesmoothing);
% eyeY = sgolayfilt(eyeY, 3, eyesmoothing);


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


%%
iTrial = iTrial + 1;
disp(iTrial)
% iTrial = 56;
% iTrial = 167;
iTrial = 178;
% 18, 56
% for iTrial = 1:numel(validTrials)
        
fprintf('%d/%d\t', iTrial, numel(validTrials))
        
thisTrial = validTrials(iTrial);
eyeix = eyeTime > tstart(iTrial) & eyeTime < tstop(iTrial);

if strcmp(Exp.D{thisTrial}.PR.name, 'BackImage')
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
plot(eyeX(eyeix), eyeY(eyeix), 'c.')
end


figure(2); clf
ax = axes();

t0 = tstart(iTrial);
plot(eyeTime(eyeix)-tstart(iTrial), eyeX(eyeix), 'k', 'Linewidth', 1.5); hold on
plot(eyeTime(eyeix)-tstart(iTrial), eyeY(eyeix), 'Color', .5*[1 1 1], 'Linewidth', 1.5);

        
if strcmp(Exp.D{thisTrial}.PR.name, 'BackImage')

    fixix = find(fixon > tstart(iTrial) & fixon < tstop(iTrial));
    nfix = numel(fixix);

    for ifix = 1:nfix
        x = [fixon(fixix(ifix))*[1 1] sacon(fixix(ifix))*[1 1]];
        x = x - t0;

        yd = ylim;
        yd = [-10 10];
        y = [yd fliplr(yd)];


        fill(x,y,'r', 'FaceAlpha', .1, 'EdgeColor', 'none')
        %     fill(*[1 1], ylim, 'r')
    end

    spkix = Exp.osp.st > tstart(iTrial) & Exp.osp.st < tstop(iTrial);
    st = Exp.osp.st(spkix)-t0;
    clu = Exp.osp.clu(spkix);
    cids = unique(clu);
    nspikes = sum(clu == cids');
    good = nspikes > 10 & nspikes < 600;
    goodix = ismember(clu, cids);
    clu = clu(goodix);
    st = st(goodix);

    [i, j] = find(clu == cids(good)');


    plot.raster(st(i), max(j)-j+20, 1, 'k', 'Linewidth', 1)
    axis off
    plot([0 1], [0 0]-15, 'k')
    plot(max(st)+[0 0] + .5, [0 50]+20, 'k')
    plot.fixfigure(gcf, 8, [5 5])
else
    ylim([-1.5 1.5])
    xlabel('Time (s)')
    ylabel('Degrees')
    plot(xlim, -.5*[1 1], 'r--')
    plot(xlim, .5*[1 1], 'r--')
    xlim([0 max(eyeTime(eyeix)-tstart(iTrial))])
    plot.fixfigure(gcf, 8, [2 1])
end


figdir = '~/Dropbox/Work/Grant_Applications/NIH_R01_Brain_2022/';
saveas(gcf, fullfile(figdir, [Exp.D{thisTrial}.PR.name 'example.pdf']))

%%

[stat, plotmeta] = fixrate_by_stim(Exp, 'stimset', {'Gabor', 'BackImage'}, 'debug', true);

%%
figure(1); clf
set(gcf, 'DefaultAxesColorOrder', lines)
cc = cc + 1;
lags = stat.(stim).lags;
% 22, 23, 45, 49
% cc = 49;
stim = 'BackImage';
cmap = lines;
h(1) = plot(lags, stat.(stim).meanRate(cc,:)); hold on
plot.errorbarFill(lags, stat.(stim).meanRate(cc,:), 2*stat.(stim).sdRate(cc,:)/sqrt(stat.(stim).numFix), 'k', 'FaceColor', cmap(1,:), 'FaceAlpha', .5, 'EdgeColor', 'none')

stim = 'Gabor';
h(2) = plot(lags, stat.(stim).meanRate(cc,:), 'Color', cmap(3,:));
plot.errorbarFill(lags, stat.(stim).meanRate(cc,:), 2*stat.(stim).sdRate(cc,:)/sqrt(stat.(stim).numFix), 'k', 'FaceColor', cmap(3,:), 'FaceAlpha', .5, 'EdgeColor', 'none')
title(cc)
xlim([-.05 .2])
xlabel('Time from fixation onset (s)')
ylabel('Spikes/Sec')
legend(h, {'Natural Image', 'Gabor Noise'})
%%
nfun = @(x) x ./ max(x, [], 2);
nfun = @(x) x ./ mean(x, 2);

figure(1); clf
n = mean(stat.BackImage.meanRate,2)/2 + mean(stat.Gabor.meanRate,2)/2;
% nfun = @(x) x; %./ n;
% [~, ind] = sort(stat.(stim).peakloc);

stims = {'Gabor', 'BackImage'};
for ii = 1:2
    stim = stims{ii};
    subplot(1,2,ii)

%     x = nfun(stat.(stim).meanRate(ind,:));
    x = nfun(stat.(stim).meanRate);
    nnmf(x)
%     ix = max(x(:,[1:100 200:600]),[],2) < 1.4;
%     ix = ix & 
%     ix = (1:size(x,1))' < 50; 
%     ix = true(size(x,1),1);
    
    imagesc(x)
%     plot(lags, mean(x)); hold on
end
%     set(gcf, 'DefaultAxesColorOrder', parula(sum(ix)))
%     imagesc(lags, 1:sum(ix), x(ix,:)); hold on
% plot([0 0], [1 sum(ix)], 'r')
%%
plot3(lags, 1:sum(ix), x(ix,:)'); hold on
% plot(xlim, 1, 'k')

%%
nfun = @(x) x ./ mean(x, 2);
x = nfun(stat.(stim).meanRate);
ix = sum(x,2) > 0;
ix = max(x(:,[1:100 200:600]),[],2) < 1.4;
[u,s,v] = svd(cov(x(ix,:)));

x(ix,:)
sd = diag(s);
nn = 3;

figure(1); clf
subplot(1,2,1)
plot(sd, '.k'); hold on
for ii = 1:nn
plot(ii, sd(ii), '.', 'Color', cmap(ii,:))
end
xlabel('Eigenvector')
ylabel('Eigenvalue')
set(gca, 'xscale', 'log')
subplot(1,2,2)

% ./sd(1:nn)'
plot(lags, u(:,1:nn))
xlabel('Time from Fixation (s)')
plot.fixfigure(gcf, 10, [4 2])
saveas(gcf, fullfile(figdir, 'sacpca.pdf'))

%%
figure(2); clf
plot(nfun(stat.(stim).meanRate(cids,:))')

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