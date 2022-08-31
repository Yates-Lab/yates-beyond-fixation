%%
% This script will only run from computers connected to the mitchell lab
% server. You need to have access to the raw data

% make sure JRCLUST is in the path first
addpath C:\Users\Jake\Documents\MATLAB\JRCLUST\

sessId = 5; % #5 is 7312017 Ellie: good example
[Exp, S] = io.dataFactoryGratingSubspace(sessId, 'spike_sorting', 'jrclustwf');

fprintf('%d single units\n', sum(Exp.osp.cgs==2))

info = io.loadEphysInfo(S.rawFilePath);

%% Set file paths and load up the JRC cluster data

fl_ = dir(fullfile(S.rawFilePath, '_processed*'));
if isempty(fl_)
    fl_ = dir(fullfile(S.rawFilePath, '_shank*'));
end

fname = fullfile(fl_(1).folder, fl_(1).name, 'ephys-jr.prm');

assert(exist(fname, 'file')==2, 'JRCLUST Param file does not exist')

hCfg = jrclust.Config(fname);
hJRC = JRC(hCfg);
if isempty(hJRC.res)
    hJRC.loadFiles();
end

%%

figDir = 'C:\Users\Jake\Dropbox\FreeViewingImported\Figures';
hCfg.siteMap = 5:16;


stim = 'Grating';
trialIdx = io.getValidTrials(Exp, stim);

iTrial = trialIdx(220);

figure(1); clf
ax0 = axes('Position', [.06 .1 .9 .8]);
plot.plotTrialEphys(Exp, hCfg, hJRC, info, iTrial);
xlim([-1 25])

xwin = [0 0.1]+18;
ywin = hCfg.siteMap([1 end])*200 + [-100 100];

% plot window
plot(xwin, ywin(1)*[1 1], 'k--')
plot(xwin, ywin(2)*[1 1], 'k--')
plot(xwin(1)*[1 1], ywin, 'k--')
plot(xwin(2)*[1 1], ywin, 'k--')
set(ax0, 'TickDir', 'out')
set(ax0, 'YTick', '')
yd = ylim;
ylim(yd + [-200 200])
plot([0 1], yd(1)*[1 1] - 100, 'k', 'Linewidth', 5)
axis off

% plot ephys in window as inset
ax = axes('Position', [.8 .1 .15 .8]);
set(gcf, 'currentaxes', ax)
plot.plotTrialEphys(Exp, hCfg, hJRC, info, iTrial);

plot(xwin, ywin(1)*[1 1], 'k--')
plot(xwin, ywin(2)*[1 1], 'k--')
plot(xwin(1)*[1 1], ywin, 'k--')
plot(xwin(2)*[1 1], ywin, 'k--')

xlim(xwin)
ylim(ywin)
title('')
axis off
set(gcf, 'Renderer', 'Painters')
set(gcf, 'PaperSize', [8 4], 'PaperPosition', [0 0 8 4])
exname = strrep(Exp.FileTag, '.mat', '');
saveas(gcf, fullfile(figDir, sprintf('%s_%d.png', exname, iTrial)));

%% plot one unit
singleUnits = find(strcmp(hJRC.hClust.clusterNotes,'single'));

selected = singleUnits(6);
iSite = hJRC.hClust.clusterSites(selected);

addpath Analysis\manuscript_freeviewingmethods\support\

nSites = min(hCfg.nSitesFigProj, size(hCfg.siteNeighbors, 1)); % by request

% center sites around cluster center site
if nSites < size(hCfg.siteNeighbors, 1)
    projSites = iSite:iSite + nSites - 1;
    if projSites(end) > max(hCfg.siteMap) % correct for overshooting
        projSites = projSites - max(projSites) + max(hCfg.siteMap);
    end
else
    projSites = sort(hCfg.siteNeighbors(:, iSite), 'ascend');
end

siteMask = ismember(hJRC.hClust.spikeSites, projSites);

dispFeatures = getFigProjFeatures(hJRC.hClust, projSites, selected);

[bgFeatures, bgTimes] = getFigTimeFeatures(hJRC.hClust, iSite); % plot background
[fgFeatures, fgTimes, YLabel] = getFigTimeFeatures(hJRC.hClust, iSite, selected(1), 1:32); % plot primary selected cluster
    

figure(1); clf
% plot(dispFeatures.bgXData, dispFeatures.bgYData, '.', 'Color', .5*[1 1 1]); hold on
% plot(dispFeatures.fgXData, dispFeatures.fgYData, '.', 'k')
plot(bgTimes/60, bgFeatures, '.', 'Color', .5*[1 1 1], 'MarkerSize', 2); hold on
plot(fgTimes/60, fgFeatures, '.', 'Color', 'k', 'MarkerSize', 2)

xlabel('Minutes')

stimG = {'BackImage', 'Gabor', 'Grating', 'Dots', 'CSD'};
cmap = lines;
mn = min(bgFeatures);
nStim = numel(stimG);
nTrials = zeros(nStim,1);
legHandle = zeros(nStim,1);
for iStim = 1:nStim
    stim = stimG{iStim};
    validTrials = io.getValidTrials(Exp, stim);
    for iTrial = 1:numel(validTrials)
        trstart = Exp.ptb2Ephys(Exp.D{validTrials(iTrial)}.STARTCLOCKTIME)/60;
        trend = Exp.ptb2Ephys(Exp.D{validTrials(iTrial)}.ENDCLOCKTIME)/60;
        if iTrial==1
            legHandle(iStim) = fill([trstart trstart trend trend], [0 mn mn 0], 'k', 'FaceColor', cmap(iStim,:), 'FaceAlpha', .5, 'EdgeColor', 'none');
        else
            fill([trstart trstart trend trend], [0 mn mn 0], 'k', 'FaceColor', cmap(iStim,:), 'FaceAlpha', .5, 'EdgeColor', 'none');
        end
    end
    nTrials(iStim) = numel(validTrials);
end

legend(legHandle(nTrials>0), stimG(nTrials>0), 'Location', 'NorthEast', 'Box', 'off')
mxt = max(bgTimes/60);

set(gca, 'TickDir', 'out', 'Box', 'off')
set(gcf, 'Renderer', 'Painters')
set(gcf, 'PaperSize', [8 4], 'PaperPosition', [0 0 8 4])
exname = strrep(Exp.FileTag, '.mat', '');
ylabel('\muVpp')
saveas(gcf, fullfile(figDir, sprintf('%s_Unit%d.pdf', exname, selected)));

%% plot amplitude of all single unit over time

singleUnits = find(strcmp(hJRC.hClust.clusterNotes,'single'));
nUnits = numel(singleUnits);

nBins = 100;
timeBins = linspace(0, max(bgTimes), nBins);
dpUnit = nan(nBins, nUnits);
seUnit = nan(nBins, nUnits);

dpBg = nan(nBins, nUnits);
seBg = nan(nBins, nUnits);
for iUnit = 1:nUnits

    selected = singleUnits(iUnit);

iSite = hJRC.hClust.clusterSites(selected);

addpath Analysis\manuscript_freeviewingmethods\support\

nSites = min(hCfg.nSitesFigProj, size(hCfg.siteNeighbors, 1)); % by request

% center sites around cluster center site
if nSites < size(hCfg.siteNeighbors, 1)
    projSites = iSite:iSite + nSites - 1;
    if projSites(end) > max(hCfg.siteMap) % correct for overshooting
        projSites = projSites - max(projSites) + max(hCfg.siteMap);
    end
else
    projSites = sort(hCfg.siteNeighbors(:, iSite), 'ascend');
end

siteMask = ismember(hJRC.hClust.spikeSites, projSites);

dispFeatures = getFigProjFeatures(hJRC.hClust, projSites, selected);

[bgFeatures, bgTimes] = getFigTimeFeatures(hJRC.hClust, iSite); % plot background
[fgFeatures, fgTimes, YLabel] = getFigTimeFeatures(hJRC.hClust, iSite, selected(1), 1:32); % plot primary selected cluster
    
    for iBin = 1:nBins-1
        bgIx = getTimeIdx(bgTimes, timeBins(iBin), timeBins(iBin+min(1, nBins-iBin)));
        muBg = mean(bgFeatures(bgIx));
        sdBg = std(bgFeatures(bgIx));
        
        fgIx = getTimeIdx(fgTimes, timeBins(iBin), timeBins(iBin+min(1, nBins-iBin)));
        muFg = mean(fgFeatures(fgIx));
        sdFg = std(fgFeatures(fgIx));
        
        dpUnit(iBin, iUnit) = muFg; %(muFg-muBg) / sqrt( (sdBg^2 + sdFg^2)/2);
        seUnit(iBin, iUnit) = sdFg/sqrt(sum(fgIx));
        
        dpBg(iBin, iUnit) = muBg; %(muFg-muBg) / sqrt( (sdBg^2 + sdFg^2)/2);
        seBg(iBin, iUnit) = sdBg/sqrt(sum(fgIx));
    end

end




figure(1); clf
for cc = 1:nUnits
    t = timeBins(1:end-1);
    x = dpUnit(1:end-1,cc);
    y = seUnit(1:end-1,cc);
    good = ~(isnan(x) | isnan(y));
%     plot.errorbarFill(t(good), x(good), y(good), 'k', 'FaceColor', cmap(cc,:), 'FaceAlpha', .5, 'EdgeColor', 'none'); hold on
    errorbar(t/60, x, y, 'Color', cmap(cc,:)); hold on
    
    
%     plot(timeBins/60, dpUnit(:,cc), 'Color', cmap(cc,:))
end

% errorbar(timeBins/60, median(dpBg,2), median(seBg,2), 'Color', .5*[1 1 1]); hold on

ylabel('Spike Amplitude')
xlabel('Time (min)')
set(gcf, 'Renderer', 'Painters')
set(gcf, 'PaperSize', [4 4], 'PaperPosition', [0 0 4 4])
exname = strrep(Exp.FileTag, '.mat', '');
ylabel('\muVpp')
saveas(gcf, fullfile(figDir, sprintf('%s_UnitsOverTime.pdf', exname)));
%%
jrc('manual', fname)

%%

jrc('preview', fname)

%%
jrc('detect-sort', fname)

%%

jrc('traces', fname)