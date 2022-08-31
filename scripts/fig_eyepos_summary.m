figDir = 'Figures/manuscript_freeviewing';

fid = 1; % print to command window

clear S;
addpath Analysis/manuscript_freeviewingmethods/

fprintf(fid, '******************************************************************\n');
fprintf(fid, '******************************************************************\n');
fprintf(fid, 'Statistics for (Eye position / saccades)\n\n\n')
%% Run main analysis on each session
fprintf(fid, '******************************************************************\n');
fprintf(fid, '******************************************************************\n');
fprintf(fid, '\t\tRUNNING EYE POSITION ANALYSIS ON All SESSIONS\n');
fprintf(fid, '******************************************************************\n');
fprintf(fid, '******************************************************************\n');
clear S
for sessId = 1:57
    fprintf(fid, '******************************************************************\n');
    fprintf(fid, 'Session %d\n', sessId)
    Exp = io.dataFactoryGratingSubspace(sessId);
      
    S(sessId) = sess_eyepos_summary(Exp);
end

%%
stim = 'All'; % loop over stimuli
monkey = 'All'; % loop over monkeys

nTrials = arrayfun(@(x) x.(stim).nTrials, S);

switch monkey
    case {'Ellie', 'E'}
        monkIx = arrayfun(@(x) strcmp(x.exname(1), 'e'), S);
        Exp = io.dataFactoryGratingSubspace(5);
    case {'Logan', 'L'}
        monkIx = arrayfun(@(x) strcmp(x.exname(1), 'l'), S);
        Exp = io.dataFactoryGratingSubspace(56);
    case {'Milo', 'M'}
        monkIx = arrayfun(@(x) strcmp(x.exname(1), 'm'), S);
    case {'All'}
        monkIx = true(size(nTrials));
end
        
good = (nTrials > 0) & monkIx;
nGood = sum(good);

bins = S(1).(stim).positionBins;
posCnt = arrayfun(@(x) x.(stim).positionCount, S(good), 'uni', 0);

dims = size(posCnt{1});
X=reshape(cell2mat(posCnt), [dims, nGood]);

figure(2); clf
C = log10(imgaussfilt(sum(X,3), .5));
h = imagesc(bins, bins, C); axis xy
colormap(plot.viridis(50))
colorbar

% plot screen width
w = Exp.S.screenRect(3)/Exp.S.pixPerDeg;
h = Exp.S.screenRect(4)/Exp.S.pixPerDeg;

hold on
plot([-w/2 w/2], [h/2 h/2], 'r', 'Linewidth', 2)
plot([-w/2 w/2], -[h/2 h/2], 'r', 'Linewidth', 2)
plot([-w/2 -w/2], [-h/2 h/2], 'r', 'Linewidth', 2)
plot([w/2 w/2], [-h/2 h/2], 'r', 'Linewidth', 2)
xlabel('Horizontal Eye Position (d.v.a.)')
ylabel('Horizontal Eye Position (d.v.a.)')
title(sprintf('Monkey: %s', monkey))

plot.fixfigure(gcf, 10, [5 4])
saveas(gcf, fullfile(figDir, sprintf('fig_eyeposition_monk%s.pdf', monkey)))

% stim = 'All';
totalDuration = arrayfun(@(x) x.(stim).totalDuration, S(good))/60;

% --- plot fixation duration
figure(4); clf
bins = S(1).(stim).fixationDurationBins * 1e3;
cnt = cell2mat(arrayfun(@(x) x.(stim).fixationDurationCnt, S(good)', 'uni', 0));
cnt = cnt ./ sum(cnt,2);
x = mean(cnt);
s = std(cnt) / sqrt(nGood);

ix = find(bins <= 100);
fill([bins(ix) bins(ix(end))], [x(ix) x(1)], 'r', 'FaceColor', .8*[1 1 1], 'EdgeColor', 'none'); hold on
ix = find(bins >= 100);
fill([bins(ix(1)) bins(ix) bins(ix(end))], [0 x(ix) x(1)], 'r', 'FaceColor', .5*[1 1 1], 'EdgeColor', 'none');
cmap=lines;
plot.errorbarFill(bins, x, 2*s, 'b', 'FaceColor', cmap(1,:), 'EdgeColor', 'none'); hold on
plot(bins, x, 'Color', cmap(1,:))

m = nan(nGood,1);
for i = 1:nGood
    [ux, ia] = unique(cumsum(cnt(i,:))./sum(cnt(i,:)));
    m(i) = interp1(ux,bins(ia), .5);
end

s = std(cnt) / sqrt(nGood);
plot.errorbarFill(bins, x, 2*s); hold on
plot(bins, x, 'k')
plot(mean(m)*[1 1], ylim, 'r', 'Linewidth', 2)
% xx = prctile(m, [25 75]);
% yy = ylim;
% fill(xx([1 1 2 2]), yy([1 2 2 1]), 'r', 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha', .5)


ylabel('Probability')
xlabel('Fixation Duration (ms)')
ylim([0 0.05])
set(gca, 'XTick', 0:250:1000, 'YTick', 0:.025:.05, 'TickDir', 'out', 'Box', 'off')
plot.fixfigure(gcf, 10, [3 3])
saveas(gcf, fullfile(figDir, sprintf('fig_fixationdur_monk%s.pdf', monkey)))

% --- Plot saccade amplitude
figure(3); clf
bins = S(1).(stim).sacAmpBinsBig;
cnt = cell2mat(arrayfun(@(x) x.(stim).sacAmpCntBig, S(good)', 'uni', 0));
% cnt = cnt ./ sum(cnt,2);
cnt = cnt ./ totalDuration(:);

x = mean(cnt);

m = nan(nGood,1);
for i = 1:nGood
    [ux, ia] = unique(cumsum(cnt(i,:))./sum(cnt(i,:)));
    try
    m(i) = interp1(ux,bins(ia), .5);
    end
end

s = std(cnt) / sqrt(nGood);
plot.errorbarFill(bins, x, 2*s); hold on
plot(bins, x, 'k')
xx = prctile(m, [25 75]);
yy = ylim;
fill(xx([1 1 2 2]), yy([1 2 2 1]), 'r', 'FaceColor', 'r', 'EdgeColor', 'none', 'FaceAlpha', .5)
ylabel('Saccade Rate (Count / Minute)')
xlabel('Amplitude (d.v.a.)')
set(gca, 'XTick', 0:5:15, 'TickDir', 'out', 'Box', 'off')

% get microsaccade plot
bins = S(1).(stim).sacAmpBinsMicro;
cnt = cell2mat(arrayfun(@(x) x.(stim).sacAmpCntMicro, S(good)', 'uni', 0));
cnt = cnt ./ totalDuration(:);

axinset = axes('Position', [.5 .5 .4 .4]);
set(gcf, 'currentaxes', axinset)

x = mean(cnt);
s = std(cnt) / sqrt(nGood);
plot.errorbarFill(bins, x, 2*s); hold on
plot(bins, x, 'k')
xlim([0 1])
ylabel('Count / Minute')
xlabel('Amplitude (d.v.a.)')
set(axinset, 'Box', 'off', 'TickDir', 'out')

% ylim([0 0.05])

plot.fixfigure(gcf, 10, [3 3])
saveas(gcf, fullfile(figDir, sprintf('fig_saccadeamp_monk%s.pdf', monkey)))



bins = S(1).(stim).DistanceBins;
cnt = cell2mat(arrayfun(@(x) x.(stim).DistanceCountCtrScreen, S(good)', 'uni', 0));
cnt = cnt ./ sum(cnt,2);

% percent within 5 d.v.a of center of screen
pc5 = sum(cnt(:,bins < 5),2) ./ sum(cnt,2);
fprintf(fid, "Monkey %s spent %02.2f %% +- %02.2f of the time < 5 d.v.a\n", monkey, mean(pc5)*100, std(pc5)*100/sqrt(nGood))
pc10 = sum(cnt(:,bins < 10),2) ./ sum(cnt,2);
fprintf(fid, "Monkey %s spent %02.2f %% +- %02.2f of the time < 10 d.v.a\n", monkey, mean(pc10)*100, std(pc10)*100/sqrt(nGood))

figure(1); clf
x = mean(cnt);
s = std(cnt) / sqrt(nGood);
plot.errorbarFill(bins, x, 2*s); hold on
plot(bins, x, 'k')
ylabel('Probability')
xlabel('Distance (d.v.a.)')
xlim([0 20])
ylim([0 .1])
set(gca, 'XTick', 0:5:20, 'YTick', 0:.05:.1, 'TickDir', 'out', 'Box', 'off')
plot.fixfigure(gcf, 10, [3 3])
saveas(gcf, fullfile(figDir, sprintf('fig_distance_monk%s.pdf', monkey)))
% imagesc(cnt)

%% Time fixating
stim = 'All';
good = true(numel(S),1);

totalDuration = arrayfun(@(x) x.(stim).totalDuration, S(good))/60;
nValidSamples = arrayfun(@(x) x.(stim).nValidSamples, S(good));
nInvalidSamples = arrayfun(@(x) x.(stim).nInvalidSamples, S(good));
nTotalSamples = arrayfun(@(x) x.(stim).nTotalSamples, S(good));
nFixationSamples = arrayfun(@(x) x.(stim).nFixationSamples, S(good));

figure(1); clf
cmap = lines;
monkeys = {'E', 'L', 'M'};
rng(555)
for m = 1:3
    ix = arrayfun(@(x) strcmpi(x.exname(1), monkeys{m}), S(good));
    n = sum(ix);
    x = nFixationSamples(ix) ./ nTotalSamples(ix) * 60;
    fprintf(fid, 'Monkey %s median fixation: %02.2f, [%02.2f, %02.2f]\n', monkeys{m}, median(x), prctile(x, 2.5), prctile(x, 97.5))
    jitter = randn(n,1);
    plot(m + .1*jitter, x, 'o', 'Color', cmap(m,:), 'MarkerFaceColor', cmap(m,:)); hold on
    plot(m + [-.2 .2], median(x)*[1 1], 'k', 'Linewidth', 2)
    
    x = nValidSamples(ix) ./ nTotalSamples(ix) * 60;
    plot(m + .1*jitter, x, 'o', 'Color', cmap(m,:)); hold on
    plot(m + [-.2 .2], median(x)*[1 1], 'k--', 'Linewidth', 2)
end

ylim([30 60])
set(gca, 'XTick', 1:3, 'XTickLabel', monkeys, 'YTick', 30:5:60, 'TickDir', 'Out', 'Box', 'off')
xlabel('Subject')
ylabel('Time (s) per minute of recording')
plot.fixfigure(gcf, 10, [4 4])
saveas(gcf, fullfile(figDir, 'fig_usableFixationTime.pdf'))


%% Percent time fixating by session

figure(2); clf
plot(nFixationSamples ./ nTotalSamples, 'o')
ylabel('Percent Time Fixating')
xlabel('Session')


%% get cumulative fixation time

good = 41:57; % sessions with fixRsvp paradigm
Sc = repmat(struct('fixtime', struct('use', false)), 57, 1);
for ex = good(:)'
    Exp = io.dataFactory(ex);
    Sc(ex) = cumulative_fix_time(Exp);
end

%% Plot cumulative fixation time
figure(1); clf
cmap = lines;

good = find(arrayfun(@(x) x.fixtime.use, Sc));
for ex = good(:)'
    plot(Sc(ex).fixtime.exTimeFor/60, Sc(ex).fixtime.fixTimeFor/60, 'Color', cmap(1,:)); hold on
    plot(Sc(ex).fixtime.exTimeFix/60, Sc(ex).fixtime.fixTimeFix/60, 'Color', cmap(5,:));
end
xd = [0 5];
plot(xd, xd, 'k--')
xlim(xd)
ylim(xd) 

plot.formatFig(gcf, [1 1], 'nature')
set(gca, 'XTick', xd, 'YTick', xd)
saveas(gcf, fullfile(figDir, 'fig_cumulativeFixTimeInset.pdf'))

% fixTimeFor = arrayfun(@(x) mean(x.fixtime.fixTimeFor(:) ./ x.fixtime.exTimeFor(:)), Sc(good));
fixTimeFix = arrayfun(@(x) mean(x.fixtime.fixTimeFix(:) ./ x.fixtime.exTimeFix(:)), Sc(good));

fixTimeFor = arrayfun(@(x) x.All.nFixationSamples/x.All.nTotalSamples, S);

fprintf('Forage prep (ddpi):\n\tTime Fixating=%02.2f [%02.2f, %02.2f] (n=%d)\n', median(fixTimeFor), prctile(fixTimeFor, [2.5 97.5]), numel(fixTimeFor))
fprintf('Fixation prep:\n\tTime Fixating=%02.2f [%02.2f,%02.2f] (n=%d)\n', median(fixTimeFix), prctile(fixTimeFix, [2.5 97.5]), numel(fixTimeFix))

figure(2); clf

xd = [0 60];
plot(xd, xd, 'k--'); hold on
plot.errorbarFill(xd, xd*mean(fixTimeFor), xd*std(fixTimeFor), 'k', 'FaceColor', cmap(1,:), 'FaceAlpha', .5, 'EdgeColor', 'none')
plot.errorbarFill(xd, xd*mean(fixTimeFix), xd*std(fixTimeFix), 'k', 'FaceColor', cmap(5,:), 'FaceAlpha', .5, 'EdgeColor', 'none')

plot(xd, (1 - 1/60)*xd, 'r') % ITI effect

% lost track
mvalid = mean(arrayfun(@(x) x.All.nValidSamples/x.All.nTotalSamples, S));
msac = mvalid*(1-mean(arrayfun(@(x) x.All.nSaccadeSamples/x.All.nValidSamples, S)));

plot(xd, xd*mvalid, 'g')
plot(xd, xd*msac, 'm')

xlabel('Experiment Time (min)')
ylabel('Cumulative Fixation Time (min)')
xlim(xd)
ylim(xd)
plot([0 5], [5 5], 'k:')
plot([5 5], [0 5], 'k:')

plot.formatFig(gcf, [2 1.8], 'nature')
saveas(gcf, fullfile(figDir, 'fig_cumulativeFixTime.pdf'))

%% Total experiment time
bnds = [0 100];
totalTime = arrayfun(@(x) x.All.totalDuration, S)/60;
fprintf('Experiment Duration = %02.2f, range = [%02.2f, %02.2f] minutes (n=%d)\n', median(totalTime), prctile(totalTime, bnds), numel(S))


hasStim = arrayfun(@(x) isfield(x.Dots, 'totalDuration'), S);
totalTime = arrayfun(@(x) x.Dots.totalDuration, S(hasStim))/60;
fprintf('Dot-mapping Duration = %02.2f, range [%02.2f, %02.2f] minutes (n=%d)\n', median(totalTime), prctile(totalTime, bnds), sum(hasStim))

hasStim = arrayfun(@(x) isfield(x.Grating, 'totalDuration'), S);
totalTime = arrayfun(@(x) x.Grating.totalDuration, S(hasStim))/60;
fprintf('Grating Duration = %02.2f, range = [%02.2f, %02.2f] minutes (n=%d)\n', median(totalTime), prctile(totalTime, bnds), sum(hasStim))


figure(2); clf
plot(totalTime, 'o')
