function Exp = fix_mitchelllab_exports(Exp)
% Function to post-process a MarmoV5 dataset
% There are several steps to postprocess the dataset:
% 1. re-synchronize clocks: Jude's lab does not use function handles to
%                           synchronize so we do that here
% 2. Find good blocks:      
%       During free-viewing, animals can look off the screen, blink, or get
%       drowsy, leading to dropped tracks. We need to find contiguous
%       blocks that are > than a minimum size. Here, we default to 500ms
%       blocks as a minimum.
% 3. Upsample / Fix missing samples from eye traces
%       The ddpi is pretty brittle and will drop samples during saccades.
%       If there are fewer than N samples of missing data, we interpolate
% 4. Detect saccades and fit a sigmoid to every pair of fixations
%       This is used for analyses that compare measured eye position to
%       pretend stable fixations

if isfield(Exp, 'Exp')
    Exp = Exp.Exp;
end

%% synchronize clocks
Exp.ptb2Ephys = synchtime.sync_ptb_to_ephys_clock(Exp, [], 'mode', 'linear', 'debug', false);
Exp.vpx2ephys = synchtime.sync_vpx_to_ephys_clock(Exp, [], 'mode', 'linear', 'debug', false);

% useful function handle
dxdt = @(x) imgaussfilt(filter([1; -1], 1, x), 5);

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

velthresh = 10;
thresh = 1.5; % size of error that can be tolerated
verbose = 1;
debug = false;
framelen = 3;

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


%% package the new analyses
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
Exp.vpx.smo = [new_timestamps, eyeDegX, eyeDegY, ones(size(eyeDegX)), dx, dy, hypot(dx, dy)];
Exp.vpx.fix = fixEye;
Exp.vpx.raw0 = Exp.vpx.raw;
Exp.vpx.raw = [new_timestamps, new_EyeX, new_EyeY, new_Pupil, bad];

% add labels
dx = dxdt(Exp.vpx.fix(:,1))/1e-3;
dy = dxdt(Exp.vpx.fix(:,2))/1e-3;
spd = hypot(dx, dy);
velthresh = 10;
Exp.vpx.Labels = 4*ones(size(dx));
Exp.vpx.Labels(spd > velthresh) = 2;
Exp.vpx.Labels(spd < velthresh) = 1;


%% detect saccades
[Exp.slist, ~] = saccadeflag.flag_saccades(Exp.vpx.smo,...
    'VFactor', 7, ...
    'MinDuration', 10, ...
    'MinGap', 20, ...
    'FlagCurve', 1.2, ...
    'SampleRate', 1000);

%% Fix 0-indexed spike cluster ids
if min(Exp.osp.clu)==0
    Exp.osp.clu = Exp.osp.clu + 1;
    Exp.osp.cids = unique(Exp.osp.clu(:))';
end



