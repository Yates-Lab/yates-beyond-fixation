function [spkS,W] = get_visual_units(Exp, varargin)
% spkS = get_visual_units(Exp, varargin)

ip = inputParser();
ip.addParameter('plotit', false)
ip.addParameter('visStimField', 'BackImage')
ip.addParameter('ROI', [-100 -100 100 100])
ip.addParameter('binSize', 10)
ip.addParameter('waveforms', nan)
ip.addParameter('numTemporalBasis', 5)
ip.parse(varargin{:});

% get waveforms
if isempty(ip.Results.waveforms)
    W = io.get_waveform_stats(Exp.osp);
else
    W = ip.Results.waveforms;
end

if ip.Results.plotit && ~isnan(W)
    figure(66); clf
    plotWaveforms(W)
    title(strrep(Exp.FileTag(1:end-4), '_', ' '))
end

%% measure visual drive
plotit = ip.Results.plotit;

%%
cids = Exp.osp.cids;
NC = numel(cids);

% trial starts
tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D));

% start / stop / start / stop
bEdges = reshape([tstart tstop]', [], 1);

% calculate firing rate during trials and during ITIs
NT = numel(bEdges)-1;
Robs = zeros(NT, NC);
for cc = 1:NC
    Robs(:,cc) = histcounts(Exp.osp.st(Exp.osp.clu==cids(cc)), bEdges);
end

% convert to firing rate
Robs = Robs ./ (bEdges(2:end)-bEdges(1:end-1));

stimSets = {'Gabor', 'Dots', 'BackImage'};

spkS = [];


%%
for cc = 1:NC

    stimFr = Robs(1:2:end-1,cc); % firing rate during trial
    isiFr = Robs(2:2:end,cc); % firing rate during ITI
    % compare images to gray screens as measure of visual drive


    Stmp = struct();
    Stmp.cid = cids(cc);
    for iStim = 1:numel(stimSets)
        validTrials = io.getValidTrials(Exp, stimSets{iStim});
        Stmp.(stimSets{iStim}) = struct('sig', nan, 'stimFr', mean(stimFr(validTrials)), 'isiFr', mean(isiFr(validTrials)));
        if numel(validTrials) > 10
            Stmp.(stimSets{iStim}).sig = ismodulated(stimFr, isiFr, validTrials);
            
        end
    end

    validTrials = io.getValidTrials(Exp, ip.Results.visStimField); % we know we show this every time

    sfr = stimFr(validTrials(1:end-1));
    ifr = isiFr(validTrials(1:end-1));
    isviz = Stmp.BackImage.sig;
    if isnan(isviz)
        isviz = false;
    end

    gtrials = validTrials;
    isiX = isiFr(validTrials(1:end-1));

    % stability analysis
    [ipoints, ~] = findchangepts(isiX, 'Statistic', 'mean', 'MinDistance', 1, 'MaxNumChanges', 2);

    n=numel(isiX);
    ipoints = [0; ipoints; n]; %#ok<AGROW>
    stableFr = [];
    for ichng = 2:numel(ipoints)
        i0 = ipoints(ichng-1)+1;
        i1 = ipoints(ichng);
        iix = i0:i1;
        stableFr = [stableFr mean(isiX(iix))*ones(size(iix))]; %#ok<AGROW>
    end

    if rsquared(isiX, stableFr) > .2 % changepoints explain 10% more variance

        len = diff(ipoints);
        [~, bigi] = max(len);

        goodix = gtrials(ipoints(bigi)+1):gtrials(ipoints(bigi+1));
    else
        goodix = gtrials(1):gtrials(n);
    end

    Stmp.stableIx = goodix;
    
    %%
    
    sptimes = Exp.osp.st(Exp.osp.clu==cids(cc));
    nbins = 200;
    K = ccg(sptimes, sptimes, nbins, 1e-3);
    nspks = K(nbins + 1);
    K(nbins+1) = 0;
    mrate = mean(K(2:20));
    Stmp.isiViolations = K(nbins+2);
    Stmp.isiCtr = nspks;
    Stmp.isiRate = Stmp.isiViolations / mrate;

    if plotit
        figure(1); clf


        subplot(2,4,2, 'align') % autocorrelation
        plot(K, '-o', 'MarkerSize', 2)
        hold on
        mrate = mean(K(1:nbins));
        plot(xlim, mrate*[1 1], 'r')
        plot(nbins+2, K(nbins+2), 'or')
        Stmp.isiViolations = K(nbins+2);
        Stmp.isiRate = Stmp.isiViolations / mrate;

        subplot(2,4,3, 'align') % natural images vs. ITI
        plot(ifr, sfr, '.'); hold on
        xlabel('Firing Rate (ITI)')
        ylabel('Firing Rate (Image)')
        plot(xlim, xlim, 'k')

        subplot(2,3,4:6) % firing rate over time (for stability)
        cmap = lines;
        plot(stimFr, 'Color', [cmap(1,:) .5]); hold on
        plot(isiFr, 'Color', [cmap(2,:) .5])
        clear h
        h(1) = plot(imboxfilt(stimFr, 11), 'Color', cmap(1,:));
        h(2) = plot(imboxfilt(isiFr, 11), 'Color', cmap(2,:));

        title(cc)
        h(3) = plot(goodix, mean(isiFr(goodix))*ones(size(goodix)), 'g');
        plot(validTrials(1)*[1 1], ylim, 'r--')
        legend(h, {'Trial', 'ITI', 'Good Trials'}, 'Location', 'Best', 'Box', 'off')


        fprintf('%d good trials\n', numel(goodix))

    end

    spkS = [spkS; Stmp]; %#ok<AGROW>
end


function [isviz,pval,stats] = ismodulated(stimFr, isiFr, validTrials)
sfr = stimFr(validTrials(1:end-1));
ifr = isiFr(validTrials(1:end-1));
% paired ttest
[isviz, pval, stats] = ttest(sfr, ifr, 'tail', 'right', 'alpha', 0.05/numel(sfr)); % scale alpha by the number of trials (more trials, more conservative)
