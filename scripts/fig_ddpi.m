

%% Load data
close all
sessId = 'logan_20200304';
spike_sorting = 'kilowf';
[Exp, S] = io.dataFactory(sessId, 'spike_sorting', spike_sorting, 'cleanup_spikes', 0);

eyePosOrig = Exp.vpx.smo(:,2:3);

% correct eye-position offline using the eye position measurements on the
% calibration trials if they were saved
try
    eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);
    
    lam = .5; % mixing between original eye pos and corrected eye pos
    Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);
end

%% plot some fixation trials

validTrials = io.getValidTrials(Exp, 'FixRsvpStim');
if numel(validTrials) > 1
    n = cellfun(@(x) numel(x.PR.NoiseHistory(:,1)), Exp.D(validTrials)); % trial length
    bad = n < 100;
    validTrials(bad) = [];
    
    % fixation starts / ends
    tstart = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(1), Exp.D(validTrials)));
    tend = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(end,1), Exp.D(validTrials)));
    n(bad) = [];
else
    fprintf("No FixRsvpStim Trials in this dataset\n")
end

%% plot individual trials
if numel(validTrials) > 1
    eyetime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
    iTrial = 0;
    figure(2); clf
    iTrial = iTrial + 1;
    if iTrial > numel(n)
        iTrial = 1;
    end
    
    iTrial = 9;
    [~, ~, idstart] = histcounts(tstart(iTrial), eyetime);
    [~, ~, idend] = histcounts(tend(iTrial), eyetime);
    
    inds = (idstart - 100):(idend + 100);
    tt = eyetime(inds) - tstart(iTrial);
    
    eyesmoothing = 19;
    eyeX = Exp.vpx.smo(inds,2);
    eyeX = sgolayfilt(eyeX, 1, eyesmoothing);
    eyeY = Exp.vpx.smo(inds,3);
    eyeY = sgolayfilt(eyeY, 1, eyesmoothing);
    plot(tt, eyeX*60, 'k'); hold on
    plot(tt, eyeY*60, 'Color', .5*[1 1 1])
    axis tight
    ylim([-1 1]*60)
    xlabel('Time (s)')
    ylabel('Arcmin')
    plot.fixfigure(gcf, 8, [4 1.5])
    title(iTrial)
%     plot(xlim, [1 1]*30, 'r--')
%     plot(xlim, -[1 1]*30, 'r--')
else
    fprintf("No FixRsvpStim Trials in this dataset\n")
end

ylim([-1 1]*40)

saveas(gcf, 'Figures/manuscript_freeviewing/fig_ExampleFixation.pdf')

%% Loop over sessions, get distribution
eyesmoothing = 19;

% get valid sessions with DDPI eye tracker
meta_file = fullfile(fileparts(which('addFreeViewingPaths')), 'Data', 'datasets.xls');
data = readtable(meta_file);

sessids = intersect( find(strcmp(data.Chamber, 'V1')) , find(strcmp(data.Eyetracker, 'DDPIv1')));
sessids(1:13) = []; % remove sessions that don't have FixRsvpStim

bins = linspace(-1, 1, 100);
nsess = numel(sessids);

Fstat = repmat(struct('sessid', [], 'trials', [], 'bins', bins, 'cnt', []), nsess,1);

for isess = 1:nsess
    
    [Exp, S] = io.dataFactory(sessids(isess));
    
    eyePosOrig = Exp.vpx.smo(:,2:3);
    
    % correct eye-position offline using the eye position measurements on the
    % calibration trials if they were saved
    try
        eyePos = io.getCorrectedEyePos(Exp, 'plot', false, 'usebilinear', false);
        
        lam = .5; % mixing between original eye pos and corrected eye pos
        Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);
    end
    
    validTrials = io.getValidTrials(Exp, 'FixRsvpStim');
    if numel(validTrials) > 1
        n = cellfun(@(x) numel(x.PR.NoiseHistory(:,1)), Exp.D(validTrials)); % trial length
        bad = n < 100;
        validTrials(bad) = [];
        
        % fixation starts / ends
        tstart = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(1), Exp.D(validTrials)));
        tend = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(end,1), Exp.D(validTrials)));
        n(bad) = [];
    else
        fprintf("No FixRsvpStim Trials in this dataset\n")
    end
    
    eyetime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
    
    eyeX = Exp.vpx.smo(:,2);
    eyeX = sgolayfilt(eyeX, 1, eyesmoothing);
    eyeY = Exp.vpx.smo(:,3);
    eyeY = sgolayfilt(eyeY, 1, eyesmoothing);
    ix = getTimeIdx(eyetime, tstart, tend);
    ix = ix & hypot(eyeX,eyeY) < 1;
    cnt = histcounts2(eyeX(ix), eyeY(ix), bins, bins);
    
%     Fstat(isess).eyeX = eyeX(ix);
%     Fstat(isess).eyeY = eyeY(ix);
    Fstat(isess).cnt = cnt;
    stime = Exp.vpx2ephys(Exp.slist(:,1));
    trial = repmat(struct('time', [], 'x', [], 'y', []), numel(validTrials),1);
    for itrial = 1:numel(validTrials)
        [~, ~, idstart] = histcounts(tstart(itrial), eyetime);
        [~, ~, idend] = histcounts(tend(itrial), eyetime);
        
        inds = (idstart - 100):(idend + 100);
        tt = eyetime(inds) - tstart(itrial);
        
        eyeX = Exp.vpx.smo(inds,2);
        eyeX = sgolayfilt(eyeX, 1, eyesmoothing);
        eyeY = Exp.vpx.smo(inds,3);
        eyeY = sgolayfilt(eyeY, 1, eyesmoothing);
        trial(itrial).trial_start = tstart(itrial);
        trial(itrial).time = tt;
        trial(itrial).x = eyeX;
        trial(itrial).y = eyeY;
        trix = stime > tstart(itrial) & stime < tend(itrial);
        trial(itrial).saccades = Exp.slist(trix,:);
        start_ind = 100;
        stop_ind = find(hypot(eyeX(start_ind:end), eyeY(start_ind:end))>1, 1);
        if isempty(stop_ind)
            stop_ind = numel(eyeX);
        else
            stop_ind = start_ind + stop_ind;
        end
        
        trial(itrial).start_ind = start_ind;
        trial(itrial).stop_ind = stop_ind-10;
        
    end
    
    
    Fstat(isess).sessid = strrep(Exp.FileTag, '.mat', '');
    Fstat(isess).trial = trial;
    
    
    
end

%% plot fixation scatter

iix = find(arrayfun(@(x) ~isempty(x.cnt), Fstat));

%%

cnt = 0;
for i = iix(:)'
    cnt = cnt + Fstat(i).cnt;
end

cnt = imgaussfilt(cnt, 1);
cnt = cnt ./ max(cnt(:));

figure(1); clf
[xx,yy] = meshgrid( (bins(1:end-1)+mean(diff(bins))/2) * 60);
contourf(xx, yy, cnt, 8, 'EdgeColor', 'w'); hold on
plot.plotellipse([0 0], [1 0; 0 1], 60, 'r--');
plot(xlim, [0 0], 'k--')
plot([0 0], ylim, 'k--')
% cmap = parula;
% cmap(1,:) = [1 1 1];
cmap = flipud([repmat(linspace(0,1,256)', 1, 2) ones(256,1)]);
colormap(cmap)
colorbar

set(gca, 'XTick', -60:30:60, 'YTick', -60:30:60)
xlim([-1 1]*60)
ylim([-1 1]*60)
xlabel('Horizontal (arcmin)')
ylabel('Vertical position (arcmin)')

plot.fixfigure(gcf, 7, [2.5 2])
saveas(gcf, 'Figures/manuscript_freeviewing/fig_FixationDistribution.pdf')

%%

figure(1); clf
nsac = [];
fdur = [];

for isess = 1:numel(Fstat)
    for itrial = 1:numel(Fstat(isess).trial)
        nsac = [nsac; size(Fstat(isess).trial(itrial).saccades, 1)];
        fdur = [fdur; diff(Fstat(isess).trial(itrial).time([1 end]))];
    end
end
% plot(Fstat(isess).trial(itrial).time, Fstat(isess).trial(itrial).x, 'b'); hold on
% plot(Fstat(isess).trial(itrial).time, Fstat(isess).trial(itrial).y, 'r');
% 
h = histogram(nsac./fdur, 50,'FaceColor', .5*[1 1 1], 'EdgeColor', 'none', 'DisplayStyle', 'bar', 'Normalization', 'probability');
xlabel('Miscrosaccade / Sec')
ylabel('Proportion')

plot.fixfigure(gcf, 7, [2.5 2])
saveas(gcf, 'Figures/manuscript_freeviewing/fig_MicrosacRate.pdf')

fprintf('Microsaccade rate:\nmedian = %02.2f [%02.2f, %02.2f]\n', median(nsac./fdur), bootci(1000, @median, nsac./fdur))


%%
validTrials = io.getValidTrials(Exp, 'FixCalib');
thisTrial = validTrials(1);



tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));

% %%
% ii = ii + 1;
% figure(1); clf
% for tt = 1:numel(Fstat(ii).trial)
%     
%     tinds = Fstat(ii).trial(tt).start_ind:Fstat(ii).trial(tt).stop_ind;
%     
%     plot(Fstat(ii).trial(tt).time(tinds), hypot(Fstat(ii).trial(tt).x(tinds), Fstat(ii).trial(tt).y(tinds))); hold on
%     
% end

%%

ttime = cell2mat(arrayfun(@(x) cell2mat(arrayfun(@(y) y.time(y.start_ind:y.stop_ind), x.trial(:), 'uni', 0)), Fstat(iix), 'uni', 0)); 
eyeX = cell2mat(arrayfun(@(x) cell2mat(arrayfun(@(y) y.x(y.start_ind:y.stop_ind), x.trial(:), 'uni', 0)), Fstat(iix), 'uni', 0));
eyeY = cell2mat(arrayfun(@(x) cell2mat(arrayfun(@(y) y.y(y.start_ind:y.stop_ind), x.trial(:), 'uni', 0)), Fstat(iix), 'uni', 0));


dist = hypot(eyeX, eyeY);

m = nanmedian(dist);
bootdist = prctile(dist(randi(numel(dist), [numel(dist) 500])), 50);
ci = prctile(bootdist, [2.5 97.5]);

fprintf('median distance from center: %02.5f [%02.5f,%02.5f]\n arcminutes', 60*m, 60*ci(1), 60*ci(2))
bins = (0:.1:1);
plot(bins, mean(dist < bins))

