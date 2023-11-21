function ddist = get_drift_distribution(Exp, varargin)
% ddist = get_drift_distribution(Exp, varargin)


ip = inputParser();
ip.addParameter('stimulusSet', 'BackImage');
ip.parse(varargin{:})

stimulusSet = ip.Results.stimulusSet;
validTrials = io.getValidTrials(Exp, stimulusSet);

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

% fixation times
fixon = Exp.vpx2ephys(Exp.slist(1:end-1,2));
sacon = Exp.vpx2ephys(Exp.slist(2:end,1));
% 
% bad = (fixon+win(1)) < min(Exp.osp.st) | (fixon+win(2)) > max(Exp.osp.st);
% fixon(bad) = [];
% sacon(bad) = [];

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

n = cellfun(@(x) numel(x), drifts);
bad = n==0 | n > 600;
drifts(bad) = [];

nFix = numel(drifts);
fprintf("%d good fixations\n", nFix)
n = cellfun(@(x) numel(x), drifts);
mx = max(n);


drift = nan(nFix, mx);
for ifix = 1:nFix
    drift(ifix,1:n(ifix)) = drifts{ifix};
end

[~, ind] = sort(n);
figure(2); clf
imagesc(drift(ind,:))

nbins = 60;
ddist = zeros(mx, nbins);
for i = 1:mx
    ddist(i,:) = histcounts(drift(:,i), 0:nbins);
end