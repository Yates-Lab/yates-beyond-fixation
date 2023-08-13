

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
winsize = 7; % Number of consecutive nans to constitute a bad segment
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
bad = interp1(raw(:,1), double(isnan(raw(:,2))), new_timestamps);

figure(1); clf
plot(new_timestamps, new_EyeX)
hold on
plot(raw(:,1), raw(:,2))


figure(2); clf
plot(new_EyeX); hold on
bads = (filter(ones(winsize,1), 1, bad)==winsize) | filter([1; -1], 1, new_timestamps)>.1;
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

%% convert eye position to degrees

[eyeDegX,eyeDegY] = io.convert_raw2dva(Exp, new_timestamps, new_EyeX, 1-new_EyeY, 1/new_binsize);

%% find all physiologically implausible saccades and exclude them
framelen = 3;
dacc = .5;

% get derivative
dxdt = @(x) imgaussfilt(filter([1; -1], 1, x), 5);

if framelen > 1
    exx = sgolayfilt(eyeDegX, 1, framelen);
    eyy = sgolayfilt(eyeDegY, 1, framelen);
else
    exx = eyeDegX;
    eyy = eyeDegY;
end


[ipoints, resid] = findchangepts(hypot(exx, eyy),'Statistic','mean','MinThreshold',5*var(hypot(exx, eyy)));

figure(1); clf
x = hypot(exx, eyy);
plot(x); hold on
plot(ipoints, x(ipoints), 'o')

%% calculate saccade peaks

spd = hypot(dxdt(exx), dxdt(eyy))/new_binsize;
acc = fix(dxdt(spd)./dacc);

peaks = findZeroCrossings(acc, -1);

figure(1); clf
plot(spd, 'k'); hold on
plot(peaks, spd(peaks), 'o')

%% step through peaks and find the best start and stop to minu
fixX = nan*exx;
fixY = nan*eyy;
buffer = 20;
peaks(diff(peaks)<buffer)=[];

debug = false;

sigmoid = @(params, x) params(1) + (params(2)-params(1))./(1 + exp( (x - params(3))/params(4)));
% sse = @(x,y) sum( (x(:)-y(:)).^2 );
fun = @(params, x) [sigmoid(params(1:4), x) sigmoid(params(5:end), x)];

N = numel(peaks)-1;
for ipeak = 2:N
    if mod(ipeak, 100)==0
        fprintf('%d/%d\n', ipeak, N)
    end

    stop = 1;
    start = 1;
    
    ix1 = (peaks(ipeak-stop)+buffer):(peaks(ipeak)-1); % previous fixation
    ix2 = (peaks(ipeak)+start):(peaks(ipeak+1)-buffer); % next fixation
    iix = min(ix1):max(ix2);

    d00 = max(sqrt((exx(iix) - mean(exx(iix))).^2 + (eyy(iix) - mean(eyy(iix))).^2));

    % initialize x parameters
    p1x = mean(exx(ix1));
    p2x = mean(exx(ix2));
    p3x = peaks(ipeak);
    p4x = var(exx(iix));
    if p2x > p1x || (p2x-p1x) < 0
        p4x = -p4x;
    end
    
    % initialize y parameters
    p1y = mean(eyy(ix1));
    p2y = mean(eyy(ix2));
    p3y = peaks(ipeak);
    p4y = var(eyy(iix));
    if p2y > p1y || (p2y-p1y) < 0
        p4y = -p4y;
    end
    
    params0x = [p1x, p2x, p3x, p4x];
    params0y = [p1y, p2y, p3y, p4y];
    params0 = [params0x params0y];
    evalc("paramsHat = lsqcurvefit(fun, params0, iix', [exx(iix) eyy(iix)])");

    if debug
        figure(1); clf
        subplot(1,2,1)
        plot(iix, exx(iix), 'k')
        hold on
        plot(iix, sigmoid(params0x, iix))
        plot(iix, sigmoid(paramsHat(1:4), iix))
        subplot(1,2,2)
        plot(iix, eyy(iix), 'k')
        hold on
        plot(iix, sigmoid(paramsHat(5:end), iix))

        pause
    end

    fixX(iix) = sigmoid(paramsHat(1:4), iix);
    fixY(iix) = sigmoid(paramsHat(5:end), iix);
    
end

disp("Done")

%%

figure(1); clf
plot(exx, 'k'); hold on
plot(fixX, 'r')

%%
clf
thresh = 2.5;
dist = hypot(exx-fixX, eyy-fixY);
starts = find(diff(dist<thresh)==1);
stops = find(diff(dist<thresh)==-1);
if starts(1) > stops(1)
    starts = [1; starts];
end

if starts(end)>stops(end)
    stops = [stops; numel(dist)];
end

good = stops-starts > 500;
starts = starts(good);
stops = stops(good);


plot(exx, 'k'); hold on
plot()
for i = 1:numel(stops)
    fill([starts(i)*[1 1] stops(i)*[1 1]],[ylim fliplr(ylim)], 'k', 'FaceAlpha', .25, 'EdgeColor', 'none')
end



%%
figure(1); clf

params0 = [p1x, p2x, p3x, p4x];
plot(iix, sigmoid(params0, iix))


%% step through peaks and find the best start and stop to minu
fixX = nan*exx;
fixY = nan*eyy;
buffer = 20;
peaks(diff(peaks)<buffer)=[];

debug = true;

for ipeak = 2:numel(peaks)


    stop = 1;
    start = 1;
    
    ix1 = (peaks(ipeak-stop)+buffer):(peaks(ipeak)-1); % previous fixation
    ix2 = (peaks(ipeak)+start):(peaks(ipeak+1)-buffer); % next fixation
    iix = min(ix1):max(ix2);
%     iix = (peaks(ipeak-1) + buffer):(peaks(ipeak+1)-buffer); % total range

    d00 = max(sqrt((exx(iix) - mean(exx(iix))).^2 + (eyy(iix) - mean(eyy(iix))).^2));

    % calculate error at starting
    fixX_ = nan*exx(iix);
    fixY_ = nan*eyy(iix);


    if debug
        figure(1); clf
        plot(iix, exx(iix), 'k'); hold on
        plot(ix1, exx(ix1), '.')
        plot(ix2, exx(ix2), '.')
        pause
    end


    ii = ix1-min(iix)+1;
    fixX_(ii) = mean(exx(ix1));
    fixY_(ii) = mean(eyy(ix1));

    ii = ix2-min(iix)+1;
    fixX_(ii) = mean(exx(ix2));
    fixY_(ii) = mean(eyy(ix2));

    fixX_ = repnan(fixX_, 'spline');
    fixY_ = repnan(fixY_, 'spline');

    % max distance
    ds = [];
    d0 = max(sqrt((fixX_ - exx(iix)).^2 + (fixY_ - eyy(iix)).^2));

    while true

        % re-initialize
        fixX_ = nan*exx(iix);
        fixY_ = nan*eyy(iix);

        % increment step away from peak
        stop = stop + 1;
        start = start + 1;

        % recompute trajectory
        ix1 = (peaks(ipeak-1)+buffer):(peaks(ipeak)-stop); % previous fixation
        ix2 = (peaks(ipeak)+start):(peaks(ipeak+1)-buffer); % next fixation

        ii = ix1-min(iix)+1;
        fixX_(ii) = mean(exx(ix1));
        fixY_(ii) = mean(eyy(ix1));

        ii = ix2-min(iix)+1;
        fixX_(ii) = mean(exx(ix2));
        fixY_(ii) = mean(eyy(ix2));

        fixX_ = repnan(fixX_, 'spline');
        fixY_ = repnan(fixY_, 'spline');

        % find max distance
        d1 = max(sqrt((fixX_ - exx(iix)).^2 + (fixY_ - eyy(iix)).^2));

        if debug
            % draw
            figure(1); clf
            plot(iix, hypot(exx(iix), eyy(iix)), 'k'); hold on
            plot(iix, hypot(fixX_, fixY_), '-r')
            title(sprintf("Max Distance %.2f", d1))
            pause(.01)
        end

        ds = [ds; d1];

        if d1 > d0
            if numel(ds)==1
                ds = [ds; d1]; % double
            end
            break
        end

        d0 = d1;


    end

    % undo last step
    stop = stop - 1;
    start = start - 1;

    % recompute trajectory
    ix1 = (peaks(ipeak-1)+buffer):(peaks(ipeak)-stop); % previous fixation
    ix2 = (peaks(ipeak)+start):(peaks(ipeak+1)-buffer); % next fixation

    ii = ix1-min(iix)+1;
    fixX_(ii) = mean(exx(ix1));
    fixY_(ii) = mean(eyy(ix1));

    ii = ix2-min(iix)+1;
    fixX_(ii) = mean(exx(ix2));
    fixY_(ii) = mean(eyy(ix2));

    fixX_ = repnan(fixX_, 'spline');
    fixY_ = repnan(fixY_, 'spline');

    if debug
        figure(1); clf
        plot(iix, exx(iix), 'k'); hold on
        plot(iix, fixX_, 'r')
        title(sprintf("Max Distance %.2f", ds(end-1)))

        figure(2); clf
        plot(ds)
        pause
    end

    fixX(iix) = fixX_;
    fixY(iix) = fixY_;
end







%%
ipeak = 1; %ipeak + 1;
win = -50:50;
iix = peaks(ipeak) + win;
prev = peaks(ipeak) - 100;


figure(1); clf
plot(win, spd(iix)); hold on
x = spd(iix);
x(x < .1*max(x)) = 0;
plot(win, x)
[params, fun] = fit_gaussian_poly(win, x);
plot(win, fun(params, win))
plot(-3*params(2)*[1 1], ylim, 'r--')
plot(3*params(2)*[1 1], ylim, 'r--')

%%

sactmp = saccadeflag.find_artifacts(new_timestamps, exx, eyy, 'order', 5, 'velthresh', 5, 'debug', true);

%%
fixX = nan*exx;
fixY = nan*eyy;

for isac = 1:numel(sactmp.tstart)-1

    iix = sactmp.endIndex(isac):sactmp.startIndex(isac+1);
    fixX(iix) = mean(exx(iix));
    fixY(iix) = mean(eyy(iix));
end

disp('Done')


fixX = repnan(fixX, 'spline');
fixY = repnan(fixY, 'spline');



figure(1); clf
plot(eyeDegX); hold on
plot(fixX)
xlim([0 800])
%%
err = hypot(eyeDegX - fixX, eyeDegY - fixY);
figure(2); clf
plot(err)

%%
iblock = iblock + 1;

dacc = .5;
framelen = 11;
goodix = blocks(iblock,1):blocks(iblock,2);

tt = new_timestamps(goodix,1);
exx = sgolayfilt(eyeDegX(goodix), 1, framelen);
eyy = sgolayfilt(eyeDegY(goodix), 1, framelen);


findchangepts(hypot(exx, eyy),'Statistic','mean','MinThreshold',5*var(hypot(exx, eyy)))


%%
spd = hypot(dxdt(exx), dxdt(eyy))/new_binsize;
acc = fix(dxdt(spd)./dacc);

peaks = findZeroCrossings(acc, -1);
figure(1); clf
% plot(acc); hold on
% plot(fix(acc))
% 
% %%
plot(spd, 'k'); hold on
plot(peaks, spd(peaks), 'o')
%%
Vthresh = 5;

figure(1); clf
plot(spd, 'k'); hold on
fixs = spd < Vthresh;
fixstart = find(diff(fixs)==1);
fixstop = find(diff(fixs)==-1);
if fixstart(1) > fixstop(1)
    fixstart = [1; fixstart];
end

if fixstop(end) < fixstart(end)
    fixstop = [fixstop; numel(spd)];
end

plot(xlim, Vthresh*[1 1], 'r')

bad = find(fixstop-fixstart < 20);
fixstart(bad+1) = [];
fixstop(bad) = [];

%%
figure(2); clf

plot(exx, 'k')
hold on

plot(eyy, 'Color', .5*[1 1 1])


nfix = numel(fixstart);
fixNum = nan*exx;
fixX = nan*exx;
fixY = nan*eyy;

for ifix = 1:nfix
    iix = fixstart(ifix):fixstop(ifix);
    fixX(iix) = mean(exx(iix));
    fixY(iix) = mean(eyy(iix));
    fixNum(iix) = ifix;

    plot(iix, exx(iix), '.k')
    plot(iix, eyy(iix), '.', 'Color', .5*[1 1 1])
end

ifix = 0;
%%

fixstop_ = fixstop;
fixstart_ = fixstart;
mse = @(x,y) mean((x-y).^2);

figure(1); clf
ifix = ifix + 1; 

iix = fixstart(ifix):fixstop(ifix+1);

fixX_ = nan*exx(iix);
fixY_ = nan*exx(iix);

% loop over two fixations and fill in
for i = 1:2 
    t0 = fixstart(ifix+i-1)-iix(1)+1;
    t1 = fixstop(ifix+i-1)-iix(1)+1;
    fixX_(t0:t1) = mean(exx(iix(t0:t1)));
    fixY_(t0:t1) = mean(eyy(iix(t0:t1)));
end

fixX_ = repnan(fixX_, 'spline');
fixY_ = repnan(fixY_, 'spline');

plot(exx(iix), 'k'); hold on
plot(fixX_)

d0 = max(hypot(fixX_-exx(iix), fixY_-eyy(iix)));
title(sprintf('Distance = %.2f',d0))
ds = d0;

while true
    fixX_ = nan*exx(iix);
    fixY_ = nan*exx(iix);

    fixstart(ifix+1) = fixstart(ifix+1) -1;
%     fixstop(ifix) = fixstop(ifix) + 1;
    % loop over two fixations and fill in
    for i = 1:2 
        t0 = fixstart(ifix+i-1)-iix(1)+1;
        t1 = fixstop(ifix+i-1)-iix(1)+1;
        fixX_(t0:t1) = mean(exx(iix(t0:t1)));
        fixY_(t0:t1) = mean(eyy(iix(t0:t1)));
    end
    
    fixX_ = repnan(fixX_, 'spline');
    fixY_ = repnan(fixY_, 'spline');
    
    plot(fixX_)
    d1 = max(hypot(fixX_-exx(iix), fixY_-eyy(iix)));
    title(sprintf('Distance = %.2f',d1))
    drawnow
    if d1 > d0
        break
    end
    d0 = d1;
    ds = [ds; d0];
end

figure, plot(ds)

%%


fixNum(iix)

plot(fixX(iix))
% 
% % move fix stop forward
% iix = fixstart(ifix):fixstop(ifix);
%     fixX(iix) = mean(exx(iix));
% fixX_ = fixX

% fixX = repnan(fixX, 'spline');
% fixY = repnan(fixY, 'spline');
% 
% plot(fixX, 'Linewidth', 2)
% plot(fixY, 'Linewidth', 2)
% 
% plot(hypot(exx-fixX, eyy-fixY))



%     findchangepts(hypot(exx, eyy),'Statistic','mean','MinThreshold',var(hypot(exx, eyy)))

%%

figure(1); clf

plot(tt, exx, 'k', 'Linewidth', 2)
hold on
plot(tt, eyeDegX(goodix), 'k-')


plot(tt, eyy, 'Color', .5*[1 1 1], 'Linewidth', 2)
plot(tt, eyeDegY(goodix), '-', 'Color', .5*[1 1 1])

sacix = find(sactmp.tstart > tt(1)-.2 & sactmp.tend < tt(end)+.2);

for isac = sacix(:)'

    fill([sactmp.tstart(isac)*[1 1], sactmp.tend(isac)*[1 1]], [ylim fliplr(ylim)], 'k', 'FaceAlpha', .25, 'EdgeColor', 'none')
end

        
plot(tt, fixX(goodix), 'Linewidth', 2)
plot(tt, fixY(goodix), 'Linewidth', 2)
% sactmp.startIndex(1)

%%

smo = [new_timestamps, exx, eyy, ones(size(exx)), dxdt(exx), dxdt(eyy)];

[~, sactmp] = saccadeflag.flag_saccades(smo,...
    'VFactor', 7, ...
    'MinDuration', 10, ...
    'MinGap', 20, ...
    'FlagCurve', 1.2, ...
    'SampleRate', 1000);

%%
spd = hypot(dxdt(exx), dxdt(eyy));
spd2 = hypot(dxdt(eyeDegX), dxdt(eyeDegY));

%%
figure(1); clf
plot(eyeDegX); hold on
plot(exx, 'Linewidth', 2); hold on

plot(spd*20)
plot(spd2*20)


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


