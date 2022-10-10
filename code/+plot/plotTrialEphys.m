function plotTrialEphys(Exp, hCfg, hJRC, info, iTrial, varargin)
% plotTrialEphys(Exp, hCfg, hJRC, info, iTrial, varargin)

scaler = 200;
yd = [min(hCfg.siteMap*scaler) max(hCfg.siteMap*scaler)];

% trial start and stop
tstart = Exp.ptb2Ephys(Exp.D{iTrial}.STARTCLOCKTIME);
tstop = Exp.ptb2Ephys(Exp.D{iTrial}.ENDCLOCKTIME);

eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));

% eye position
eyix = eyeTime > tstart & eyeTime < tstop;
eyeX = Exp.vpx.smo(eyix,2);
eyeY = Exp.vpx.smo(eyix,3);

% saccades
sacon = Exp.vpx2ephys(Exp.slist(:,1));
sacinds = find(getTimeIdx(sacon, tstart, tstop));

sacon = sacon(sacinds);
% sacoff = Exp.vpx2ephys(Exp.slist(sacinds,2));
% sacons = Exp.vpx.smo(Exp.slist(sacinds,4),2);

% sacon = io.convertTimeToSamples(sacon, info.sampleRate, info.timestamps, info.fragments);
% sacoff = io.convertTimeToSamples(sacoff, info.sampleRate, info.timestamps, info.fragments);

eyeSamp = io.convertTimeToSamples(eyeTime(eyix), info.sampleRate, info.timestamps, info.fragments);
offset = eyeSamp(1);
bad = find(Exp.vpx.Labels(eyix)>2);
eyeX(bad) = nan;
eyeY(bad) = nan;
xax = eyeSamp-offset;
xax = xax / info.sampleRate;
plot(xax, eyeX*20 + yd(2) + 2*scaler, 'k'); hold on
plot(xax, eyeY*20 + yd(2) + 2*scaler, 'Color', .5*[1 1 1]);

cmap = lines;
cmap(2,:) = cmap(1,:);
cmap(1,:) = [1 0 0];

labels = bwlabel(diff(bad)==1);
for il = unique(labels)'
    ix = il==labels;
    ix = bad(ix);
    ix(end) = [];
    x = eyeSamp(ix)-offset;
    x = x/info.sampleRate;
    fill([x(1) x(1) x(end) x(end)], yd(2) + [0 3*scaler 3*scaler 0], 'r', 'FaceColor', cmap(2,:), 'EdgeColor', 'none', 'FaceAlpha', .5)
end

nsac = numel(sacon);

for i = 1:nsac
    sstart = Exp.slist(sacinds(i),4);
    sstop = Exp.slist(sacinds(i),5);
    tt = io.convertTimeToSamples(eyeTime(sstart:sstop), info.sampleRate, info.timestamps, info.fragments);
    tt = tt - offset;
    tt = tt / info.sampleRate;
    plot(tt, Exp.vpx.smo(sstart:sstop,2)*20 + yd(2) + 2*scaler, 'Color', cmap(1,:));
    plot(tt, Exp.vpx.smo(sstart:sstop,3)*20 + yd(2) + 2*scaler, 'Color', cmap(1,:));
end



yd(2) = yd(2) + 500;

swin = io.convertTimeToSamples([tstart tstop], info.sampleRate, info.timestamps, info.fragments);
windowBounds = swin; %[1 16e3]+ 100;


recFilename = hCfg.rawRecordings{1}; % if multiple files exist, load first
            
fprintf('Opening %s\n', recFilename);
hRec = jrclust.detect.newRecording(recFilename, hCfg);
            
% nSamplesTotal = hRec.dshape(2);            
% nTimeTraces = hCfg.nSegmentsTraces;

tracesRaw = hRec.readRawROI(hCfg.siteMap, windowBounds(1):windowBounds(2));

% bandpass filter
tracesFilt = jrclust.filters.bandpassFilter(tracesRaw', hCfg)' * hCfg.bitScaling;

tracesFilt = tracesFilt * .5;

xt = (1:max(size(tracesFilt)));
xt = xt / info.sampleRate;
plot(xt, bsxfun(@plus, double(tracesFilt)', double(hCfg.siteMap)*scaler), 'Color', .8*[1 1 1])
hold on 

spikesInRange = hJRC.hClust.spikeTimes > windowBounds(1) & hJRC.hClust.spikeTimes < windowBounds(2);

spikeTimes = hJRC.hClust.spikeTimes(spikesInRange);
spikeSites = hJRC.hClust.spikeSites(spikesInRange);
spikeClusters = hJRC.hClust.spikeClusters(spikesInRange);

% only include units in the site map
ix = ismember(spikeSites, hCfg.siteMap);
spikeTimes = spikeTimes(ix);
spikeSites = spikeSites(ix);
spikeClusters = spikeClusters(ix);

cgs = strcmp(hJRC.hClust.clusterNotes, 'single')*2 + strcmp(hJRC.hClust.clusterNotes, 'multi');
cids = unique(spikeClusters);
cids(cids==0) = [];
cgs = cgs(cids);

NC = numel(cids);
cmap = lines(NC);
sctr = 1;
for cc = 1:NC
    wf = hJRC.hClust.meanWfGlobalRaw(:,:,cids(cc));
    wft = hCfg.evtWindowRawSamp(1):hCfg.evtWindowRawSamp(2);
    
    ix = cids(cc)==spikeClusters;
    st = double(spikeTimes(ix)-windowBounds(1))';
    nspk = numel(st);
    st = reshape([st + wft(:); nan(1, nspk)], [], 1);
    st = st / info.sampleRate;
    site = mode(spikeSites(ix));
    for i = -1:1
        
        if ~ismember(site + i, hCfg.siteMap)
            continue
        end
        
        ss = reshape([double((spikeSites(ix) + i)*scaler)' + wf(:,site+i); nan(1, nspk)], [], 1);
        
        if any(ss > max(hCfg.siteMap*scaler) + 10)
            continue
        end
        
        if cgs(cc) == 2
            plot(st, ss, '-', 'Color', cmap(sctr,:));
        elseif cgs(cc)==1
            plot(st, ss, '-', 'Color', .5*[1 1 1], 'Linewidth', 1);
        end
    end
    if cgs(cc) == 2
        sctr = sctr + 1;
    end
end

ylim(yd)


exname = strrep(strrep(Exp.FileTag, '.mat', ''), '_', ' ');
title(sprintf('%s Trial: %d', exname, iTrial))