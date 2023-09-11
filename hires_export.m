
%% Load session

datadir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'hires');
outputdir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'stim_movies');

flist = dir(fullfile(outputdir, '*.hdf5'));

%% Loop over sessions and import (this will be super slow)
% sesslist = {'allen_20220610.mat'};
sesslist = {'logan_20200304.mat'};
% sesslist = {'logan_20191231.mat'};
for isess = 1

    %% Load dataset
    processedFileName = sesslist{isess};
    fname = fullfile(datadir, processedFileName);
    sessname = strrep(processedFileName, '.mat', '');

    exportexists=sum(arrayfun(@(x) contains(x.name, sessname), flist))>0;
    fprintf('%d) [%s] exported = %d\n', isess, sessname, exportexists)

    %     assert(sum(arrayfun(@(x) contains(x.name, sessname), flist))==0, 'export_hires: session already exists')

    %     assert(exist(fname, 'file'), "export_hires: preprocessed file does not exist. run step 01 first")
    %     if exportexists
    %         disp('Export exists. skipping')
    %         continue
    %     end

    Exp = load(fname);
    Exp = fix_mitchelllab_exports(Exp);

    %%

    

    %% get coarse resolution spatial RFs
    
    % these data come from the foveal representation so the central 3 d.v.a
    % are sufficien
    ROI = [-1 -1 1 1]*3;
    % coarse bin size
    binSize = .25;
    
    Frate = 120;
    eyeposexclusion = 20;
    win = [-1 20];
    ROIWINDOWSIZE = 150; % spatial dimensions of the high-res ROI

    [Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
        'ROI', ROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
        'eyePosExclusion', eyeposexclusion * Exp.S.pixPerDeg, ...
        'eyePos', Exp.vpx.fix, 'frate', Frate, ...
        'fastBinning', true, ...
        'smoothing', 2);

    % use indices only while eye position is on the screen
    scrnBnds = (Exp.S.screenRect(3:4) - Exp.S.centerPix) / Exp.S.pixPerDeg;
    scrnBnds = 1.5 * scrnBnds;
    eyePosAtFrame = opts.eyePosAtFrame/Exp.S.pixPerDeg;

    ix = (eyePosAtFrame(:,1) + ROI(1)) >= -scrnBnds(1) & ...
        (eyePosAtFrame(:,1) + ROI(3)) <= scrnBnds(1) & ...
        (eyePosAtFrame(:,2) + ROI(2)) >= -scrnBnds(2) & ...
        (eyePosAtFrame(:,2) + ROI(4)) <= scrnBnds(2);

    fprintf('%02.2f%% of gaze positions are safely on screen\n', 100*mean(ix))

    numspikes = sum(RobsSpace(ix,:));

    % forward correlation
    stas = forwardCorrelation(full(Xstim), sum(RobsSpace-mean(RobsSpace, 1),2), win, find(ix), [], true, false);

    close all
    figure(1); clf
    % stas = forwardCorrelation(Xstim, mean(RobsSpace,2), win);
    stas = stas / std(stas(:)) - mean(stas(:));
    wm = [min(stas(:)) max(stas(:))];
    nlags = size(stas,1);
    for ilag = 1:nlags
        subplot(2, ceil(nlags/2), ilag)
        imagesc(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg, reshape(stas(ilag, :), opts.dims), wm)
        title(sprintf('lag: %02.2f', ilag*16))
        axis xy
    end

    %% find ROI

    winsize = ROIWINDOWSIZE;
    rf = reshape(std(stas), opts.dims);
    rf = (rf - min(rf(:))) / (max(rf(:)) - min(rf(:)));
    [con, ar, ctr] = get_rf_contour(opts.xax, opts.yax, rf, 'thresh', .7);
    imagesc(opts.xax, opts.yax, rf);
    hold on
    plot(ctr(1), ctr(2), 'or')
    S.rect = round([ctr ctr]) + [-1 -1 1 1]*winsize/2;


    in_pixels = false;
    if in_pixels
        ppd = 1;
    else
        ppd = Exp.S.pixPerDeg;
    end

    figure(2); clf
    imagesc(opts.xax/ppd, opts.yax/ppd, rf)
    axis xy

    hold on
    plot(S.rect([1 3])/ppd, S.rect([2 2])/ppd, 'r', 'Linewidth', 2)
    plot(S.rect([1 3])/ppd, S.rect([4 4])/ppd, 'r', 'Linewidth', 2)
    plot(S.rect([1 1])/ppd, S.rect([2 4])/ppd, 'r', 'Linewidth', 2)
    plot(S.rect([3 3])/ppd, S.rect([2 4])/ppd, 'r', 'Linewidth', 2)
    title('Average Spatial RF & ROI')
    xlabel('Azimuth (pixels)')
    ylabel('Elevation (pixels)')

    plot.fixfigure(gcf, 12, [4 4]);
    saveas(gcf, fullfile('figures/hires_export', sprintf('%s_roi.pdf', strrep(processedFileName, '.mat', ''))) )
    % pixels run down so enforce this here
    S.rect([2 4]) = sort(-S.rect([2 4]));

    %% Do high-res reconstruction using PTB (has to replay the whole experiment)
    Exp.FileTag = processedFileName;
    S.spikeSorting = 'kilo';
    stimsets = {'Dots', 'Gabor', 'BackImage', 'Grating', 'FixRsvpStim', 'FixFlashGabor'};
%     fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Gabor'}, 'overwrite', false, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false, 'EyeSmoothing', 19, 'EyeSmoothingOrder', 1);
    fname = make_stimulus_file_for_py(Exp, S, 'stimlist', stimsets, 'overwrite', false, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false, 'useFix', true);
    
    %% get STAs to check that you have the right rect
    stim = 'Gabor';
    tset = 'Train';
    spike_sorting = 'kilo';
    Stim = h5read(fname, ['/' stim '/' tset '/Stim']);
    % Robs = h5read(fname, ['/' stim '/' set '/Robs']);
    ftoe = h5read(fname, ['/' stim '/' tset '/frameTimesOe']);

    frate = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'frate');
    st = h5read(fname, ['/Neurons/' spike_sorting '/times']);
    clu = h5read(fname, ['/Neurons/' spike_sorting '/cluster']);
    cids = h5read(fname, ['/Neurons/' spike_sorting '/cids']);
    sp = struct();
    sp.st = st;
    sp.clu = clu;
    sp.cids = cids;
    Robs = binNeuronSpikeTimesFast(sp, ftoe-8e-3, 1/frate);

    % Robs =
    eyeAtFrame = h5read(fname, ['/' stim '/' tset '/eyeAtFrame']);
    labels = h5read(fname, ['/' stim '/' tset '/labels']);
    NX = size(Stim,2);
    NY = size(Stim,3);
    NC = size(Robs,2);

    Stim = reshape(Stim, size(Stim, 1), NX*NY);
    Stim = zscore(single(Stim));


    %% Pick a lag and compute the STA quickly for all cells
    figname = sprintf('results_%s_4.pdf', sessname);
    lag = 8;
    ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
    ix = ecc < 5.2 & labels == 1 & (1:numel(ecc))'> lag;
    Rdelta = Robs - mean(Robs);
    Rdelta = Rdelta(ix,:);
    sta = (Stim(find(ix)-lag,:).^2)'*Rdelta;
    % sta = (Stim(find(ix)-lag,:))'*Rdelta;
    [~, ind] = sort(std(sta));

    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));
    figure(10); clf
    for cc = 1:NC
        subplot(sx, sy, cc, 'align')
        imagesc(reshape(sta(:,ind(cc)), [NX NY]))
        axis off
    end

    set(gcf, 'PaperSize', [7 7], 'PaperPosition', [0 0 7 7])
    exportgraphics(gcf, figname);

    figure(11); clf
    plot(std(sta(:,ind)), '-o'); hold on
    thresh = 200;
    cids = find(std(sta) > thresh);
    plot(xlim, thresh*[1 1], 'r')
    set(gcf, 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
    exportgraphics(gcf, figname, 'Append', true);

%% try to find center of gaze

% we need the trial starts and stops and protocol name
% we need raw eye traces
% we need meta data that we're interest in (e.g., Grating / Probe / Reward)


%% write eye position to h5 file

figure(1); clf
plot(Exp.vpx2ephys(Exp.vpx.smo(:,1)), Exp.vpx.smo(:,2)); hold on
plot(ftoe, (eyeAtFrame(:,2)-Exp.S.centerPix(1))/Exp.S.pixPerDeg, '.')

%%
Stat = struct();
Stat.timestamps = Exp.vpx2ephys(Exp.vpx.smo(:,1));
Stat.eyeposDeg = Exp.vpx.smo(:,2:3);
Stat.ppd = Exp.S.pixPerDeg;
Stat.ctrpx = Exp.S.centerPix;

fname = io.h5_add_struct(fname, Stat, '/ddpi');

%% write trial starts, stops, protocol, and reward

Stat = struct();

Stat.trialStarts = cellfun(@(x) x.START_EPHYS, Exp.D);
Stat.trialStops = cellfun(@(x) x.END_EPHYS, Exp.D);
nTrials = numel(Exp.D);
Stat.protocol = repmat({'unrecognized'}, nTrials, 1);
Stat.rewardTimes = Exp.ptb2Ephys(cell2mat(cellfun(@(x) x.rewardtimes', Exp.D, 'UniformOutput', false)));

stimList = {'Grating', 'Gabor', 'Dots', 'BackImage', ...
        'Forage', ...
        'FixRsvpStim', ...
        'FaceCal', ...
        'FixCalib', ...
        'ForageStaticLines', ...
        'FixFlashGabor', ...
        'MTDotMapping', ...
        'DriftingGrating'};

for i = 1:numel(stimList)
    tlist = io.getValidTrials(Exp, stimList{i});
    if isempty(tlist)
        continue
    end
    for j = tlist(:)'
        Stat.protocol{j} = stimList{i};
    end
end

fname = io.h5_add_struct(fname, Stat, '/trials');

% %%
% [m, s, bc, v, tspcnt] = eventPsth(Exp.vpx2ephys(Exp.slist(:,1)), Stat.rewardTimes, [-.5 .5], .005);
% 
% figure(1); clf
% plot(bc, m)

%% write the imagefiles and 


tlist = io.getValidTrials(Exp, 'BackImage');
imstarts = Exp.ptb2Ephys(cellfun(@(x) x.PR.startTime, Exp.D(tlist)));
imstops = Exp.ptb2Ephys(cellfun(@(x) x.PR.imageOff, Exp.D(tlist)));
imrect = cell2mat(cellfun(@(x) x.PR.destRect, Exp.D(tlist), 'uni', 0));
imlist = [];
for i = tlist(:)'
    tmp = strsplit(Exp.D{i}.PR.imagefile, '/');
    imlist = [imlist; tmp(end)];
end

Stat = struct();
Stat.startTimes = imstarts;
Stat.stopTimes = imstops;
Stat.imageList = imlist;
Stat.imageRect = imrect;
fname = io.h5_add_struct(fname, Stat, '/BackImage');

% f = h5info(fname, '/BackImage');
% arrayfun(@(x) x.Name, f.Groups(2).Datasets, 'uni', 0)


%% write the Grating condition

tlist = io.getValidTrials(Exp, 'Grating');

isfrozen = cellfun(@(x) x.PR.frozenSequence==1, Exp.D(tlist));

Stat = struct();
Stat.timestamps = Exp.ptb2Ephys(cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,1), Exp.D(tlist), 'uni', 0)));
Stat.ori = cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,2), Exp.D(tlist), 'uni', 0));
Stat.sf = cell2mat(cellfun(@(x) x.PR.NoiseHistory(:,3), Exp.D(tlist), 'uni', 0));

figure(1); clf
plot(Stat.timestamps, Stat.ori, '.')

fname = io.h5_add_struct(fname, Stat, '/Grating');

% tlist = io.getValidTrials(Exp, 'ForageStaticLines');
% 

%%
%%%%%%%%%%%%%
% TODO: clean this up a bit and then comit the changes!!


%%%%%%%

%% Copy to server
server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'; %/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
command = 'scp ';
command = [command fname ' '];
command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname)


%%



% DO THIS IN PYTHON
% 
% % find center of gaze
% % 1) get all fixation trials
% % 2) get all epochs when the monkey is fixating
% % 3) get peak of distribution and call that center
% fixTrials = find(cellfun(@(x) contains(lower(x.PR.name), 'fix'), Exp.D));
% N = numel(fixTrials);
% fprintf("Found %d trials with fixation\n", N)
% 
% trialStarts = cellfun(@(x) x.START_EPHYS, Exp.D(fixTrials));
% trialStops = cellfun(@(x) x.END_EPHYS, Exp.D(fixTrials));
% 
% idx = getTimeIdx(Exp.vpx2ephys(Exp.vpx.smo(:,1)), trialStarts, trialStops);
% 
% eyeX = Exp.vpx.smo(idx,2);
% eyeY = Exp.vpx.smo(idx,3);
% 
% bs = (1/60); % 1 arcminute
% edges = -1:bs:1;
% ctrs = edges(1:end-1) + bs/2;
% 
% C = histcounts2(eyeX, eyeY, edges, edges);
% C = imgaussfilt(C,1);
% 
% figure(1); clf
% imagesc(ctrs, ctrs, C'); hold on
% plot([0 0], ylim, 'y')
% plot(xlim, [0 0], 'y')
% axis xy
% colorbar
% 
% [yy,xx] = meshgrid(ctrs);
% 
% levels = [.25 .5 .75, .9];
% N = numel(levels);
% ctrxy = zeros(N,2);
% C = C ./ max(C(:));
% for i = 1:N
%     level = levels(i);
% 
%     iix = C > level;
%     w = C(iix) ./ sum(C(iix));
%     
%     cx = xx(iix)'*w;
%     cy = yy(iix)'*w;
%     ctrxy(i,1) = cx;
%     ctrxy(i,2) = cy;
% 
%     scatter(cx, cy, 'o', 'filled')
% end
% 
% 
% Stat = struct();
% Stat.binCenters = ctrs;
% Stat.histogram = C;
% Stat.threshold = levels;
% Stat.centerOfGaze = ctrxy;
% 
% fields = fieldnames(Stat);
% for ifield = 1:numel(fields)
%     h5create(fname,'/CenterOfGaze/binCenters', size(ctrs))
%     h5write(fname, '/CenterOfGaze/cgs', ctrs)











    %% Plot histogram of gaze positions
    xgrid = -8:1:8;
    xax = xgrid;

    eyeX = (eyeAtFrame(:,2)-Exp.S.centerPix(1))/Exp.S.pixPerDeg;
    eyeY = (eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;

    C = histcounts2(eyeX, eyeY, xax, xax);
    figure(1); clf
    subplot(1,2,1)
    imagesc(xax, xax, C')
    title('Eye At Frame')

    iix = ~isnan(hypot(Exp.vpx.smo(:,2), Exp.vpx.smo(:,3)));
    validTrials = io.getValidTrials(Exp, 'Gabor');
    vix = getTimeIdx(Exp.vpx.smo(:,1), cellfun(@(x) x.START_VPX, Exp.D(validTrials)), cellfun(@(x) x.END_VPX, Exp.D(validTrials)));
    iix = vix & iix;
    C2 = histcounts2(Exp.vpx.smo(iix,2), -Exp.vpx.smo(iix,3), xax, xax);
    subplot(1,2,2)
    imagesc(xax, xax, C2')
    title('vpx Smo')
    colormap parula

    set(gcf, 'PaperSize', [7 7], 'PaperPosition', [0 0 7 7])
    exportgraphics(gcf, figname, 'Append', true);

    %% Loop over gaze positions and re-calculate Spike-Triggered Energy, find centers

    R = Robs(:,cids);
    Rbar = mean(R);

    n = numel(xax);
    win = 1;

    figure(10); clf
    set(gcf, 'Color', 'w')
    ctrs = nan(n, n, 2);
    threshs = nan(n, n);

    axs = plot.tight_subplot(n, n, .01, .01);

    for ii = 1:n
        for jj = 1:n

            win_ = min(max(hypot(eyeX, eyeY)*win, 1), 3);
            ix = (hypot(eyeX - xax(ii), eyeY - xax(jj)) < win_) & labels == 1;
            fprintf('x: %.2f, y: %0.2f, %d samples\n', xax(ii), xax(jj), sum(ix))
            if sum(ix) < 200
                continue
            end

            ix = find(ix);
            ix = ix(ix > lag);
            Rdelta = R(ix,:) - Rbar;
            sta = (Stim(ix-lag,:).^2)'*Rdelta;
            %         sta = reshape((V(ix-lag,:,:).^2), [],size(Stim,2))'*Rdelta;

            %         subplot(n, n, (ii-1)*n + jj)
            set(gcf, 'currentaxes', axs((jj-1)*n + ii))
            I = reshape(mean(sta,2), [NX, NY]);
            I = imgaussfilt(I,3);
            I = (I - min(I(:))) / (max(I(:))- min(I(:)));
            imagesc(I); colormap gray
            axis off
            hold on
            [con, ar, ctr, thresh] = get_rf_contour(1:NX, 1:NY, I);

            if isnan(ctr(1))
                continue
            end
            ctrs(jj,ii,:) = ctr;

            plot(con(:,1), con(:,2), 'r', 'Linewidth', 2)
            threshs(jj,ii) = thresh;

            axis off
            drawnow
        end

    end


    % plot.fixfigure(gcf, 12, [7 7])
    set(gcf, 'PaperSize', [7 7], 'PaperPosition', [0 0 7 7])
    exportgraphics(gcf, figname, 'Append', true);

    %% plot map of errors
    figure(11); clf
    set(gcf, 'Color', 'w')
    subplot(1,2,1)
    imagesc(ctrs(:,:,1).*(threshs<.8))
    title('Horizontal RF center')
    xlabel('Gaze Position (d.v.a.)')
    ylabel('Gaze Position (d.v.a.)')
    subplot(1,2,2)
    imagesc(ctrs(:,:,2).*(threshs<.8))
    title('Vertical RF center')
    xlabel('Gaze Position (d.v.a.)')
    ylabel('Gaze Position (d.v.a.)')

    set(gcf, 'PaperSize', [7 5], 'PaperPosition', [0 0 7 5])
    exportgraphics(gcf, figname, 'Append', true);

    %% fit interpolant to errors
    xax = xgrid;
    [xx, yy] = meshgrid(xax);

    figure(1); clf;
    [i,j] = find(xx==0 & yy==0);
    cx = NX/2;
    cy = NY/2;

    mask = (threshs<.9);
    errX = ctrs(:,:,1)-cx;
    errY = ctrs(:,:,2)-cy;
    dX = conv2(errX, [-1 1], 'same').^2 + conv2(errX, [-1; 1], 'same').^2; imagesc(dX)
    dY = conv2(errX, [-1 1], 'same').^2 + conv2(errX, [-1; 1], 'same').^2; imagesc(dX)
    mask = dX < 20 & dY < 20 & mask;
    errX = errX.*mask;
    errY = errY.*mask;
    % errX = ctrs(:,:,1).*mask-cx;
    % errY = ctrs(:,:,2).*mask-cy;
    errYflip = ctrs(:,:,2).*mask-cy;


    plot3(xx(:), yy(:), errX(:), '.'); hold on

    % Set up fittype and options.
    ft = 'linear'; % interpolant type
    options = fitoptions(ft, 'Normalize', 'on');

    [xData, yData, zDataX] = prepareSurfaceData( xx(mask), yy(mask), errX(mask) );
    [fitresultX, gofX] = fit( [xData, yData], zDataX, ft, fitoptions);

    [xData, yData, zDataY] = prepareSurfaceData( xx(mask), yy(mask), errY(mask) );
    [fitresultY, gofY] = fit( [xData, yData], zDataY, ft, fitoptions);

    [xData, yData, zDataYflip] = prepareSurfaceData( xx(mask), yy(mask), errYflip(mask) );
    [fitresultYflip, gofYflip] = fit( [xData, -yData], zDataYflip, ft, fitoptions);

    % plot
    figure(1); clf
    subplot(1,2,1)
    h = plot( fitresultX, [xData, yData], zDataX );
    legend( h, 'Cubic Interpolation', 'z vs. x, y', 'Location', 'NorthEast', 'Interpreter', 'none' );
    % Label axes
    xlabel( 'x', 'Interpreter', 'none' );
    ylabel( 'y', 'Interpreter', 'none' );
    zlabel( 'z', 'Interpreter', 'none' );
    grid on
    view( -14.7, 11.1 );

    subplot(1,2,2)
    h = plot( fitresultY, [xData, yData], zDataY );
    legend( h, 'Cubic Interpolation', 'z vs. x, y', 'Location', 'NorthEast', 'Interpreter', 'none' );
    % Label axes
    xlabel( 'x', 'Interpreter', 'none' );
    ylabel( 'y', 'Interpreter', 'none' );
    zlabel( 'z', 'Interpreter', 'none' );
    grid on
    view( -14.7, 11.1 );
    %
    % figure(2); clf
    % h = plot( fitresultYflip, [xData, -yData], zDataYflip );
    % legend( h, 'Cubic Interpolation', 'z vs. x, y', 'Location', 'NorthEast', 'Interpreter', 'none' );
    % % Label axes
    % xlabel( 'x', 'Interpreter', 'none' );
    % ylabel( 'y', 'Interpreter', 'none' );
    % zlabel( 'z', 'Interpreter', 'none' );
    % grid on
    % view( -14.7, 11.1 );

    %%
    xHat = fitresultX(xx, yy);
    yHat = fitresultY(xx, yy);
    yHatflip = fitresultYflip(xx, -yy);

    plot3(xx(:), yy(:), xHat(:), '.')

    for i = find(mask)'
        plot3([xx(i) xx(i)], [yy(i) yy(i)], [errX(i) xHat(i)], 'k-'); hold on
    end

    figure(2); clf,
    subplot(1,2,1)
    imagesc(errX)
    title('Horizontal', 'Measured')
    subplot(1,2,2)
    imagesc(reshape(xHat, [n n]))
    title('Horizontal', 'Interpolant')

    set(gcf, 'PaperSize', [7 5], 'PaperPosition', [0 0 7 5])
    exportgraphics(gcf, figname, 'Append', true);

    figure(3); clf,
    subplot(1,2,1)
    imagesc(xax, xax, errY)
    title('Vertical', 'Measured')
    subplot(1,2,2)
    imagesc(xax, xax, reshape(yHat, [n n]))
    title('Vertical', 'Interpolant')

    set(gcf, 'PaperSize', [7 5], 'PaperPosition', [0 0 7 5])
    exportgraphics(gcf, figname, 'Append', true);

    % figure(3); clf,
    % subplot(1,2,1)
    % imagesc(reshape(yHat, [n n]), [-20 20])
    % subplot(1,2,2)
    % imagesc(reshape(yHatflip, [n n]), [-20 20])

    %% Do shifting
    [xx,yy] = meshgrid(1:NX, 1:NY);

    NT = size(Stim, 1);
    X = reshape(Stim, [NT, NX, NY]);
    shiftX = fitresultX(eyeX, eyeY);
    shiftY = fitresultY(eyeX, eyeY);
    V = zeros(size(X));

    for i = 1:NT
        V(i,:,:) = interp2(squeeze(X(i,:,:)), shiftX(i)+xx, shiftY(i)+yy, 'nearest', 0);
    end
    disp("Done")

    %% check whether shifts produce nans

    failedshift = squeeze(sum(sum(isnan(V), 2),3))>0;
    shiftmag = hypot(shiftX, shiftY);
    eyedist = hypot(eyeX, eyeY);


    figure(1); clf
    histogram(eyedist(~failedshift), 'BinEdges', linspace(0, 10, 100)); hold on
    histogram(eyedist(failedshift), 'BinEdges', linspace(0, 10, 100))
    legend({'Successful Shift', 'Failed Shift'})
    xlabel('Gaze distance from center')

    set(gcf, 'PaperSize', [4 4], 'PaperPosition', [0 0 4 4])
    exportgraphics(gcf, figname, 'Append', true);


    %%
    lag = 8;
    ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
    ix = ecc < 5.2 & labels == 1 & (1:numel(ecc))'> lag;
    Rdelta = Robs - mean(Robs);
    Rdelta = Rdelta(ix,:);
    X = reshape(V, size(Stim));
    X(isnan(X))=0;
    sta0 = (X(find(ix)-lag,:))'*Rdelta;

    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));
    figure(12); clf
    for cc = 1:NC
        subplot(sx, sy, cc, 'align')
        imagesc(reshape(sta0(:,cc), [NX NY]))
        axis off
        title(cc)
    end

    set(gcf, 'PaperSize', [7 7], 'PaperPosition', [0 0 7 7])
    exportgraphics(gcf, figname, 'Append', true);


end


%%




%%


%%

% 
% 
% %% Get spatiotemporal RFs
% NC = size(Robs,2);
% nlags = 20;
% nstim = size(Stim,2);
% ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
% R = Robs;
% Rbar = mean(R);
% 
% stas = zeros(nlags, nstim, NC);
% for lag = 1:nlags
%     fprintf('%d/%d lags\n', lag, nlags)
%     ix = ecc < 5.2 & labels == 1;
%     ix = find(ix);
%     ix = ix(ix > lag);
%     Rdelta = R(ix,:) - Rbar; %mean(R(ix,:));
%     stas(lag, :, :) = (X(ix-lag,:))'*Rdelta;
% end
% 
% cc = 0; % initialize iterator
% 
% %% plot one by one
% figure(2); clf
% cc = cc + 1;
% if cc > NC
%     cc = 1;
% end
% % cc = 58
% % cc = 81
% sta = stas(:,:,cc);
% % zscore the sta
% % sta = (sta - mean(sta(:))) ./ std(sta(:));
% 
% clim = max(abs(sta(:)));
% xax = (1:NX)/Exp.S.pixPerDeg*60;
% yax = (1:NY)/Exp.S.pixPerDeg*60;
% for ilag = 1:nlags
%    subplot(2,ceil(nlags/2), ilag, 'align')
%    imagesc(xax, yax, reshape(sta(ilag,:), [NX NY])', [-1 1]*clim)
%    title(ilag)
% end
% 
% % % colormap(plot.viridis)
% colormap(parula)
% title(cc)
% 
% 
% %%
% 
% figure(1); clf
% imagesc(reshape(mean(squeeze(std(stas,[], 1)),2), [80, 80]))
% 
% 
% 
% 
% %% check that the online measured eye position 
% figure(1); clf
% plot(eyeX); hold on
% plot(eyeAtFrame(:,5), '.')
% 
% figure(2); clf
% plot(eyeY); hold on
% plot(-eyeAtFrame(:,6), '.')
% 
% % plot(eyeX - eyeAtFrame(:,5), '.')
% %% check that you can create the same shift from the raw eye traces
% 
% % shift calculated on this (processed) dataset
% shiftX = fitresultX(eyeX, eyeY);
% shiftY = fitresultY(eyeX, eyeY);
% 
% % shift using "raw" eye pos
% shiftX2 = fitresultX(eyeAtFrame(:,5), -eyeAtFrame(:,6));
% shiftY2 = fitresultY(eyeAtFrame(:,5), -eyeAtFrame(:,6));
% 
% figure(1); clf
% xdelta = (shiftX - shiftX2);
% ydelta = (shiftY - shiftY2);
% mdx = mean(xdelta.^2, 'omitnan');
% mdy = mean(ydelta.^2, 'omitnan');
% 
% histogram(xdelta, 'BinEdges', linspace(-1, 1, 100))
% hold on
% histogram(ydelta, 'BinEdges', linspace(-1, 1, 100))
% xlabel('Difference in Shifter output (pixels)')
% 
% % histogram(xdelta, 'BinEdges', linspace(-10, 10, 100))
% 
% 
% 
% % fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Dots', 'Gabor', 'BackImage', 'Grating', 'FixRsvpStim'}, 'overwrite', false, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false, 'EyeCorrection', shifter, 'EyeSmoothing', 19, 'EyeSmoothingOrder', 1);
% fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'FixRsvpStim'}, 'overwrite', true, 'GazeContingent', true, 'includeProbe', true, 'usePTBdraw', false, 'EyeCorrection', shifter, 'EyeSmoothing', 19, 'EyeSmoothingOrder', 1);
% 
% %% Copy to server
% server_string = 'jake@bancanus'; %'jcbyts@sigurros';
% output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'; %/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';
% 
% data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
% command = 'scp ';
% command = [command fname ' '];
% command = [command server_string ':' output_dir];
% 
% system(command)
% 
% fprintf('%s\n', fname)
