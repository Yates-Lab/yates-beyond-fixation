

%% Load data

Exp = load('/Users/jake/Dropbox/MarmoLabWebsite/PSA/Allen_Reimport/Allen_2022-06-01_13-31-38_V1_64b.mat');
if isfield(Exp, 'Exp')
    Exp = Exp.Exp;
end

%% synchronize clocks
Exp.ptb2Ephys = synchtime.sync_ptb_to_ephys_clock(Exp, [], 'mode', 'linear', 'debug', false);
Exp.vpx2Ephys = synchtime.sync_vpx_to_ephys_clock(Exp, [], 'mode', 'linear', 'debug', false);

%% cleanup eye traces and re-detect saccades

% USER SET PROPERTIES
winsize = 20; % ms of consecutive nans to constitute a bad segment
padding = 20; % pad bad segments by 20 ms on each end
min_block_size = 500; % only include blocks if they are greater than 500ms
new_binsize = 1e-3; % sampleing rate after resampling

raw = Exp.vpx.raw;
duplicates = find(diff(raw(:,1))==0);
fprintf("found %d duplicates\n", numel(duplicates))

matches = mean(raw(duplicates,2:3)==raw(duplicates+1,2:3),2)==1;
confirmed_duplicates = duplicates(matches);
fprintf("removing %d duplicates\n", numel(confirmed_duplicates))
raw(confirmed_duplicates,:) = [];

duplicates = find(diff(raw(:,1))==0);

fprintf("found %d duplicates\n", numel(duplicates))

% replace nans with neighbors if possible
for id = 2:4
    raw(duplicates,id) = nanmean(reshape(raw(duplicates + [-1 0 1], id), 3, []));
end

% now, eliminate all double samples (this shouldn't do anything)
duplicates = find(diff(raw(:,1))==0);
raw(duplicates,:) = []; %#ok<FNDSB> 

duplicates = find(diff(raw(:,1))==0);

fprintf("found %d duplicates\n", numel(duplicates))

% resample

blocks = [0; find(diff(raw(:,1)) > 1); numel(raw(:,1))];
new_timestamps = [];
for i = 1:numel(blocks)-1
    tmp = (raw(blocks(i)+1,1):new_binsize:raw(blocks(i+1),1))';
    new_timestamps = [new_timestamps; tmp]; %#ok<AGROW> 
end

MODE = 'pchip';
new_EyeX = interp1(raw(:,1), repnan(raw(:,2), MODE), new_timestamps, MODE);
new_EyeY = interp1(raw(:,1), repnan(raw(:,3), MODE), new_timestamps, MODE);
new_Pupil = interp1(raw(:,1), raw(:,4), new_timestamps, MODE);
bad = interp1(raw(:,1), double(isnan(raw(:,2))), new_timestamps)>0;

% %%
% figure(1); clf
% plot(new_timestamps, new_EyeX, '-')
% hold on
% plot(raw(:,1), raw(:,2))
% plot(new_timestamps(bad), new_EyeX(bad), '.')

%% Label bad segments if they have too many consecutive missing data points
L = bwlabel(bad);
stats = regionprops(L); %#ok<MRPBW> 
cnt = arrayfun(@(x) x.Area, stats);
bads = ismember(L, find(cnt > winsize));
disp([sum(bad), sum(bads)])

figure(2); clf
plot(new_EyeX); hold on
block_end = find(diff(bads)==1);
block_start = find(diff(bads)==-1);
if bads(1)==0
    block_start = [1; block_start];
end

if bads(end)==0
    block_end = [block_end; numel(bads)];
end

block_start = block_start + padding;
block_end = block_end - padding;

blocks = [block_start block_end];

badblocks = (blocks(:,2) - blocks(:,1)) < min_block_size;

fprintf('Found %d blocks\n', size(blocks,1))

blocks = blocks(~badblocks,:);

fprintf('Found %d blocks\n', size(blocks,1))

for iblock = 1:size(blocks,1)
    fill(blocks(iblock,[1 1 2 2]), [ylim fliplr(ylim)], 'r', 'FaceColor', 'r', 'FaceAlpha', .2, 'EdgeColor', 'none')
end

plot(find(bads), new_EyeX(bads), '.')
%% convert eye position to degrees

[eyeDegX,eyeDegY] = io.convert_raw2dva(Exp, new_timestamps, new_EyeX, 1-new_EyeY, 1/new_binsize);


%% Step through blocks and fit sigmoids to pairs of fixations
iblock = 0;

velthresh = 10;
thresh = 1.5; % size of error that can be tolerated
verbose = 1;
debug = false;

% iblock = iblock + 1;
% if iblock > size(blocks,1)
%     iblock = 1;
% end
fixX = nan*eyeDegX;
fixY = nan*eyeDegY;

new_blocks = [];
for iblock = 1:size(blocks,1)
    goodix = blocks(iblock,1):blocks(iblock,2);

    [fstarts, fstops] = label_fixations(eyeDegX(goodix),eyeDegY(goodix),framelen,velthresh,new_binsize,winsize);
    fixEye2 = fit_sigmoids_to_eyetraces(eyeDegX(goodix),eyeDegY(goodix), [fstarts fstops], 0, debug);
    if all(isnan(fixEye2(:,1)))
        continue
    end
    fixEye2(:,1) = repnan(fixEye2(:,1), 'spline');
    fixEye2(:,2) = repnan(fixEye2(:,2), 'spline');
    
    fixX(goodix) = fixEye2(:,1);
    fixY(goodix) = fixEye2(:,2);

    if verbose > 0
        fprintf("Found %d fixations in block %d\n", numel(fstarts), iblock)
    end

    dist = hypot(eyeDegX(goodix) - fixEye2(:,1), eyeDegY(goodix) - fixEye2(:,2));
    % find good segments and label them
    differ = filter([1; -1], 1, dist<thresh);
    starts = find(differ==1);
    stops = find(differ==-1);
    if ~isempty(starts)

        if isempty(stops), stops = numel(dist); end
        if starts(1) > stops(1), starts = [1; starts]; end
        if starts(end)>stops(end), stops = [stops; numel(dist)]; end

        n = numel(starts);
        for i = 1:n
            block_ = goodix(1) + [starts(i) stops(i)];
            new_blocks = [new_blocks; block_];
        end
    end

    if verbose > 1
        figure(1); clf
        set(gcf, 'Color', 'w')
        subplot(3,1,1:2)
        plot(eyeDegX(goodix), 'Color', .5*[1 1 1], 'Linewidth', 2); hold on
        plot(fixEye2(:,1), 'k', 'Linewidth', 1)

        plot(eyeDegY(goodix), 'Color', [1 .5 .5], 'Linewidth', 2); hold on
        plot(fixEye2(:,2), 'Color', [1, 0, 0], 'Linewidth', 1)

        axis tight
        ylabel('Position (d.v.a.)')
        title(sprintf("Block %d", iblock))


        if ~isempty(starts)
            if starts(1) > stops(1), starts = [1; starts]; end
            if starts(end)>stops(end), stops = [stops; numel(dist)]; end

            % plot bad segments
            for i = 1:numel(starts)
                t = [starts(i)*[1 1] stops(i)*[1 1]];
                %             fill(t, [ylim, fliplr(ylim)], 'r', 'FaceAlpha', .5, 'EdgeColor', 'none')
                fill(t, [ylim, fliplr(ylim)], 'g', 'FaceAlpha', .25, 'EdgeColor', 'none')
            end
        end

        subplot(3,1,3)
        fill([0:numel(dist) numel(dist)], [0 dist' 0], 'k', 'FaceAlpha', .5); hold on
        xlabel('ms')
        ylabel('Total Error (d.v.a.)')
        axis tight
        plot(xlim, thresh*[1 1], 'r--')
        ylim([0 thresh*1.5])
        pause(0.1)
    end
end


%%

fixEye = [fixX fixY];
durations = new_blocks(:,2)-new_blocks(:,1);
good_blocks = durations > 400;
blocks_ = new_blocks(good_blocks,:);

figure(1); clf
histogram(durations*new_binsize, 'binEdges', linspace(0, 10, 1000))

idx = getTimeIdx(1:numel(fixX), blocks_(:,1), blocks_(:,2));
fixEye(~idx,:) = nan;

dx = dxdt(eyeDegX);
dy = dxdt(eyeDegY);
Exp.vpx.smo = [new_timestamps, eyeDegX, eyeDegY, ones(size(exx)), dx, dy, hypot(dx, dy)];
Exp.vpx.fix = fixEye;

%%
[slist, sactmp] = saccadeflag.flag_saccades(Exp.vpx.smo,...
    'VFactor', 7, ...
    'MinDuration', 10, ...
    'MinGap', 20, ...
    'FlagCurve', 1.2, ...
    'SampleRate', 1000);


%% TODO: 
% package the new fixation based eye traces in smo?
% then test on highres RFs

%% fit eye traces with a sigmoid for each saccade
framelen = 3;
dacc = .5;

% get derivative
dxdt = @(x) imgaussfilt(filter([1; -1], 1, x), 5);

velthresh = 10;
[fstarts, fstops] = label_fixations(eyeDegX,eyeDegY,framelen,velthresh,new_binsize,winsize);
idx = getTimeIdx(fstarts,blocks(:,1), blocks(:,2));
fstarts(~idx) = [];
fstops(~idx) = [];

[fixEye, stats] = fit_sigmoids_to_eyetraces(exx, eyy, [fstarts fstops], 0, debug);

% seperate X and Y traces
fixX = fixEye(:,1);
fixY = fixEye(:,2);

%% re-label fixations based on this trace
velthresh = 5;
dfx = dxdt(fixX)/new_binsize;
dfy = dxdt(fixY)/new_binsize;
fspd = hypot(dfx, dfy);
fixs = fspd < velthresh;

fixstarts = find(diff(fixs)==1);
fixstops = find(diff(fixs)==-1);
if fixstarts(1) > fixstops(1)
    fixstarts(1) = [1; fixstarts];
end

if fixstarts(end) > fixstops(end)
    fixstops = [fixstops; numel(fixs)];
end


%% find bad segments within valid blocks
thresh = 1.5;
dist = hypot(eyeDegX - fixX, eyeDegY - fixY);

starts = find(diff(dist>thresh)==1);
stops = find(diff(dist>thresh)==-1);
if starts(1) > stops(1)
    starts = [1; starts];
end

if starts(end)>stops(end)
    stops = [stops; numel(dist)];
end

didx = getTimeIdx(starts, blocks(:,1), blocks(:,2));
starts = starts(didx);
stops = stops(didx);
iblock = 0;

%% setp through blocks
iblock = 1;
% iblock = iblock + 1;
if iblock > size(blocks,1)
    iblock = 1;
end

didx = find(starts > blocks(iblock,1) & starts < blocks(iblock,2));

fidx = find(fixstarts > blocks(iblock,1) & fixstops < blocks(iblock,2));

fprintf("Found %d bad segments in block %d\n", numel(didx), iblock)
fprintf("Found %d fixations in block %d\n", numel(fidx), iblock)

goodix = blocks(iblock,1):blocks(iblock,2);

velthresh = 10;
[fstarts, fstops] = label_fixations(eyeDegX(goodix),eyeDegY(goodix),framelen,velthresh,new_binsize,winsize);
fixEye2 = fit_sigmoids_to_eyetraces(eyeDegX(goodix),eyeDegY(goodix), [fstarts fstops], 0, debug);


figure(1); clf
set(gcf, 'Color', 'w')
subplot(3,1,1:2)
plot(eyeDegX(goodix), 'Color', .5*[1 1 1], 'Linewidth', 2); hold on
plot(fixX(goodix), 'k', 'Linewidth', 1)
plot(fixEye2(:,1), 'b', 'Linewidth', 1)
plot(eyeDegY(goodix), 'Color', [1 .5 .5], 'Linewidth', 2); hold on
plot(fixY(goodix), 'Color', [1, 0, 0], 'Linewidth', 1)

axis tight
ylabel('Position (d.v.a.)')

% plot bad segments
for ii = 1:numel(didx)
    i = didx(ii);
    t = [starts(i)*[1 1] stops(i)*[1 1]] - goodix(1) + 1;
    t = min(t, numel(goodix));
    fill(t, [ylim, fliplr(ylim)], 'r', 'FaceAlpha', .5, 'EdgeColor', 'none')
end

subplot(3,1,3)
err = hypot(eyeDegX(goodix) - fixX(goodix), eyeDegY(goodix) - fixY(goodix));
fill([0:numel(err) numel(err)], [0 err' 0], 'k', 'FaceAlpha', .5); hold on
xlabel('ms')
ylabel('Total Error (d.v.a.)')
axis tight

%%

smo = [new_timestamps, exx, eyy, ones(size(exx)), dxdt(exx), dxdt(eyy)];

[~, sactmp] = saccadeflag.flag_saccades(smo,...
    'VFactor', 7, ...
    'MinDuration', 10, ...
    'MinGap', 20, ...
    'FlagCurve', 1.2, ...
    'SampleRate', 1000);


%%
figure(1); clf
plot(sactmp.size, sactmp.vel, '.'); hold on

badsacs = find(sactmp.vel > .2*sactmp.size);
fprintf('Found %d bad sacs\n', numel(badsacs))



%% plot block by block

for iblock = 1:size(blocks,1)
    goodix = blocks(iblock,1):blocks(iblock,2);
    
    tt = new_timestamps(goodix,1);
    exx = sgolayfilt(eyeDegX(goodix), 1, framelen);
    eyy = sgolayfilt(eyeDegY(goodix), 1, framelen);

    figure(1); clf

    plot(tt, exx, 'k', 'Linewidth', 2)
    hold on
    plot(tt, eyeDegX(goodix), 'k-')

    
    plot(tt, eyy, 'Color', .5*[1 1 1], 'Linewidth', 2)
    plot(tt, eyeDegY(goodix), '-', 'Color', .5*[1 1 1])

    sacix = find(sactmp.tstart > tt(1)-1 & sactmp.tend < tt(end)+1);
    
    for isac = sacix(:)'
        fill([sactmp.tstart(isac)*[1 1], sactmp.tend(isac)*[1 1]], [ylim fliplr(ylim)], 'k', 'FaceAlpha', .25, 'EdgeColor', 'none')
    end
        
        pause
end

%% engbert way
framelen = 21;
debug = false;
slist = [];

% get derivative
dxdt = @(x) imgaussfilt(filter([1; -1], 1, x), 5);

fprintf('Detecting Saccades Using Engbert and Mergenthaler\n ')
for iblock = 1:size(blocks,1)

    goodix = blocks(iblock,1):blocks(iblock,2);
    
    tt = new_timestamps(goodix,1);
    exx = sgolayfilt(eyeDegX(goodix), 1, framelen);
    eyy = sgolayfilt(eyeDegY(goodix), 1, framelen);
    psz = ones(size(exx));
    velx = dxdt(exx);
    vely = dxdt(eyy);
    
    smo = [tt, exx, eyy, psz, velx, vely];
    
    
    [~, sactmp] = saccadeflag.flag_saccades(smo,...
        'VFactor', 7, ...
        'MinDuration', 10, ...
        'MinGap', 20, ...
        'FlagCurve', 1.2, ...
        'SampleRate', 1000);

%     sactmp(:,1:3) = sactmp(:,1:3)/1e3;

    if isempty(sactmp)
        continue
    end

%     [startsac endsac peaksac kstartsac kendsac kpeaksac flag]]

    
    if debug
        figure(1); clf

        plot(tt, exx, 'k', 'Linewidth', 2)
        hold on
        plot(tt, eyeDegX(goodix), 'k-')

        
        plot(tt, eyy, 'Color', .5*[1 1 1], 'Linewidth', 2)
        plot(tt, eyeDegY(goodix), '-', 'Color', .5*[1 1 1])
%         nsac = size(sactmp,1);
%         for isac = 1:nsac
%             fill([sactmp(isac,1)*[1 1], sactmp(isac,2)*[1 1]], [ylim fliplr(ylim)], 'k', 'FaceAlpha', .25, 'EdgeColor', 'none')
%         end
% 
%         pause
%     end
        nsac = numel(sactmp.tstart);
        for isac = 1:nsac
            fill([sactmp.tstart(isac)*[1 1], sactmp.tend(isac)*[1 1]], [ylim fliplr(ylim)], 'k', 'FaceAlpha', .25, 'EdgeColor', 'none')
        end
        
        pause
    end

    if iblock == 1
        saccades = sactmp;
    else
        fields = fieldnames(saccades);
        for ifield = 1:numel(fields)
            field = fields{ifield};
            saccades.(field) = [saccades.(field); sactmp.(field)];
        end
    end

end

fprintf('Done. Found %d saccades\n', numel(saccades.tstart))

%% cloherty way
debug = false;

num_blocks = size(blocks,1);
dtmax = zeros(num_blocks, 1);
for iblock = 1:num_blocks

    goodix = blocks(iblock,1):blocks(iblock,2);
    
    tt = new_timestamps(goodix,1);
    dtmax(iblock) = max(diff(tt));
    exx = sgolayfilt(eyeDegX(goodix), 1, framelen);
    eyy = 1- sgolayfilt(eyeDegY(goodix), 1, framelen);
    psz = ones(size(exx));
    velx = dxdt(exx);
    vely = dxdt(eyy);
    
    
    
    sactmp = saccadeflag.find_saccades(tt, exx, eyy,...
        'Wn', [0.1,0.2], ...
        'order', 48, ...
        'accthresh', 50,...
        'velthresh', 5,...
        'velpeak', 5,...
        'isi', 0.025, ...
        'dt', 0.1, ...,
        'debug', false);

    if isempty(sactmp)
        continue
    end
    
    if debug
        figure(1); clf
        plot(tt, exx, 'k')
        hold on
        plot(tt, eyy)
        nsac = numel(sactmp.tstart);
        for isac = 1:nsac
            fill([sactmp.tstart(isac)*[1 1], sactmp.tend(isac)*[1 1]], [ylim fliplr(ylim)], 'k', 'FaceAlpha', .25, 'EdgeColor', 'none')
        end

        pause
    end

    if iblock == 1
        saccades2 = sactmp;
    else
        fields = fieldnames(saccades2);
        for ifield = 1:numel(fields)
            field = fields{ifield};
            saccades2.(field) = [saccades2.(field); sactmp.(field)];
        end
    end
end

fprintf('Done. Found %d saccades\n', numel(saccades2.tstart))

%%
field1 = 'size';
field2 = 'vel';

figure(1); clf
subplot(1,2,1)
plot(saccades.(field1), saccades.(field2)*1e3, '.'); hold on
plot(xlim, 200*xlim, 'r')
subplot(1,2,2)
plot(saccades2.(field1), saccades2.(field2), '.'); hold on
plot(xlim, 200*xlim, 'r')

%% 

iix = find(saccades.size < 1 & saccades.vel*1e3 > 2e3);
figure(1); clf
plot(new_timestamps, '.')
hold on
plot(saccades.tstart, '.')

%%
i = i + 1;
ind = find(new_timestamps > saccades.tstart(iix(i)), 1);
figure(1); clf
win = -100:100;
plot(win,  new_EyeX(ind + win) - mean(new_EyeX(ind + win)))
hold on
plot(win, new_EyeY(ind + win) - mean(new_EyeY(ind + win)))

%%

histogram(saccades.duration, 'binEdges', 0:.01:1); hold on
histogram(saccades2.duration, 'binEdges', 0:.01:1)

%%
% figure(1); clf
% plot(saccades.dX, saccades.dY, '.')

%%

% Saccade processing:
% Perform basic processing of eye movements and saccades
Exp = saccadeflag.run_saccade_detection_cloherty(Exp, ...
    'ShowTrials', false,...
    'accthresh', 2e4,...
    'velthresh', 10,...
    'velpeak', 10,...
    'isi', 0.02);

% track invalid sampls
Exp.vpx.Labels(isnan(raw(:,2))) = 4;

validTrials = io.getValidTrials(Exp, 'Grating');
for iTrial = validTrials(:)'
    if ~isfield(Exp.D{iTrial}.PR, 'frozenSequence')
        Exp.D{iTrial}.PR.frozenSequence = false;
    end
end

%%
figure(1); clf
plot.raster(Exp.osp.st, Exp.osp.clu);

hold on

iix = io.getValidTrials(Exp, 'Gabor');

estart = cellfun(@(x) x.START_EPHYS, Exp.D(iix));
estop = cellfun(@(x) x.END_EPHYS, Exp.D(iix));

for i = 1:numel(iix)

    fill([estart(i)*[1 1] estop(i)*[1 1]], [ylim, fliplr(ylim)], 'r', 'FaceAlpha', .25, 'EdgeColor', 'none')


end


