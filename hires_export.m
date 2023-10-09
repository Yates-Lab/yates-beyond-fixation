
%% Load session

datadir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'hires');
outputdir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'stim_movies');

flist = dir(fullfile(outputdir, '*.hdf5'));

figdir = fullfile('figures/hires_export');
%% Loop over sessions and import (this will be super slow)
sesslist = arrayfun(@(x) x.name, dir(fullfile(datadir, '*.mat')), 'uni', 0);
% sesslist = {'logan_20200304.mat'};
% sesslist = {'Allen_2022-02-24_12-52-27_V1_64b.mat'};
% sesslist = {'Allen_2022-02-16_12-09-11_V1_64flip_b.mat'};
isess = 1;
%%
% sesslist = {'logan_20191231.mat'};
for isess = 1:numel(sesslist)

    %% Load dataset
    processedFileName = sesslist{isess};
    fname = fullfile(datadir, processedFileName);
    sessname = strrep(processedFileName, '.mat', '');

    exportexists=sum(arrayfun(@(x) contains(x.name, sessname), flist))>0;
    fprintf('%d) [%s] exported = %d\n', isess, sessname, exportexists)

    Exp = load(fname);
    run_hires_export(Exp, 'fig_name', sessname)
end

%% Copy to server
server_string = 'jake@bancanus'; %'jcbyts@sigurros';
output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'; %/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

data_dir = getpref('FREEVIEWING', 'PROCESSED_DATA_DIR');
command = 'scp ';
command = [command fname ' '];
command = [command server_string ':' output_dir];

system(command)

fprintf('%s\n', fname)


%% Plot FixRsvpStim

idx = io.getValidTrials(Exp, 'FixRsvpStim');
num_trials = numel(idx);
stim_start = nan(num_trials, 1);
trial_start = nan(num_trials,1);
stim_end = nan(num_trials,1);
for i = 1:num_trials
    if ~isempty(Exp.D{idx(i)}.PR.NoiseHistory)
        stim_start(i) = Exp.D{idx(i)}.PR.NoiseHistory(1,1);
        stim_end(i) = Exp.D{idx(i)}.PR.NoiseHistory(end,1);
        trial_start(i) = Exp.D{idx(i)}.START_EPHYS;
    end
    
end

thisTrial = idx(1);
% figure(1); clf
% plot(trial_start-Exp.ptb2Ephys(stim_start), '.')

%%

% plot(Exp.D{thisTrial}.PR.NoiseHistory(:,1)

dur = stim_end - stim_start;
ix = dur > .5;
stim_start = stim_start(ix);
stim_end = stim_end(ix);
dur = dur(ix);

stim_start = Exp.ptb2Ephys(stim_start);
stim_end = Exp.ptb2Ephys(stim_end);

win = [-.2 max(dur)];

bin_size = 1e-3;

cids = unique(Exp.osp.clu);


NC = numel(cids);
nperfig = 20;
nfigs = ceil(NC/nperfig);

sx = ceil(sqrt(nperfig));
sy = round(sqrt(nperfig));

for cc = 1:NC
    fignum = ceil(cc/nperfig);
    figure(fignum)
    
    iplot = mod(cc, nperfig);
    if iplot==0
        iplot=nperfig;
    end
    
    sptimes = Exp.osp.st(Exp.osp.clu==cids(cc));
    [~,~,bc,~,spcnt] = eventPsth(sptimes, stim_start, win, bin_size);
    [~, ind] = sort(dur);
    [i,j] = find(spcnt(ind,:));
    subplot(sx, sy, iplot)
    plot.raster(bc(j),i); hold on
    plot(dur(ind), 1:numel(ind))
    xlim(win)
    title(cc)
end



%%
% cc = 32;
% cc = 0;
cc = cc + 1;
if cc > NC
    cc = 1;
end

figure
sptimes = Exp.osp.st(Exp.osp.clu==cids(cc));
bin_size= 1/240;
[~,~,bc,~,spcnt] = eventPsth(sptimes, stim_start, win, bin_size);




[~, ind] = sort(dur);
[i,j] = find(spcnt(ind,:));

plot.raster(bc(j),i); hold on
plot(dur(ind), 1:numel(ind))
xlim(win)
title(cc)

figure(10); clf
plot(mean(spcnt))



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

