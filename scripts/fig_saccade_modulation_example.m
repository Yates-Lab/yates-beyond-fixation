

%% Load analyses
sesslist = io.dataFactory;
sesslist = sesslist(1:57); % exclude monash sessions

% Spatial RFs
sfname = fullfile('Data', 'spatialrfs.mat');
load(sfname)

% Grating RFs
fittype = 'loggauss';
gfname = fullfile('Data', sprintf('gratrf_%s.mat', fittype));
load(gfname)

% FFT RF
fftname = fullfile('Data', 'fftrf.mat');
load(fftname)

% Saccade-triggered rates
fixname = fullfile('Data', 'fixrate.mat');
load(fixname)

% Waveforms
wname = fullfile('Data', 'waveforms.mat');
load(wname)


%% Explore

iEx = 45;

fprintf('Loading session [%s]\n', sesslist{iEx})
Exp = io.dataFactory(sesslist{iEx}, 'spike_sorting', Sgt{iEx}.sorter);
[rf_post, plotmeta] = fixrate_by_fftrf(Exp, Srf{iEx}, Sgt{iEx}, 'debug', false, 'plot', false, 'usestim', 'post', 'alignto', 'fixon');

%% plot it
         
fig = figure(3); clf
fig.Position = [100 100 800 400];

cc = 17;

fixdur = plotmeta.fixdur;
[~, ind] = sort(fixdur);
[i,j] = find(squeeze(plotmeta.spks(ind,cc,:)));

m = 4;
n = 8;
layout1 = tiledlayout(m,n);
layout1.TileSpacing = 'compact';
layout1.Padding = 'compact';

% -- RASTER
ax = nexttile([2 2]);
             
f1 = find(fixdur(ind) > .20,1);
f2 = find(fixdur(ind) > .50,1);
ix = i >= f1 & i <= f2;

plot.raster(lags(j(ix)), i(ix), 5); axis tight
hold on
cmap = lines;
plot(fixdur(ind(f1:f2)), f1:f2, '-', 'Color', cmap(4,:))
% plot([0 0], ylim, 'Color', cmap(4,:))

xlabel('Time from fixation (s)')
ylabel('Fixation #')
ax.YTick = [];
ax.Box = 'off';
ax.TickDir = 'out';
ax.TickLength = ax.TickLength*2;

% -- MEAN FIRING RATE BY STIMULUS
ax = nexttile(3, [2 2]);

% BackImage
lags = fixrat{iEx}.BackImage.lags;
m = fixrat{iEx}.BackImage.meanRate(cc,:);
s = 2*fixrat{iEx}.BackImage.sdRate(cc,:) / sqrt(fixrat{iEx}.BackImage.numFix);
plot.errorbarFill(lags, m, s, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', 'none', 'FaceAlpha', .5); hold on
h = plot(lags, m, 'Color', cmap(1,:));

% Grating
lags = fixrat{iEx}.Grating.lags;
m = fixrat{iEx}.Grating.meanRate(cc,:);
s = 2*fixrat{iEx}.Grating.sdRate(cc,:) / sqrt(fixrat{iEx}.Grating.numFix);
plot.errorbarFill(lags, m, s, 'k', 'FaceColor', cmap(4,:), 'EdgeColor', 'none', 'FaceAlpha', .5); hold on
h(2) = plot(lags, m, 'Color', cmap(4,:));

xlim(lags([1 end]))
xlabel('Time From Fixation (s)')
ylabel('Firing Rate (sp s^{-1})')
ax.Box = 'off';
ax.TickDir = 'out';
ax.TickLength = ax.TickLength*2;
legend(h, {'Natural Image', 'Flashed Grating'}, 'Location', 'Best')

% -- FIXATIONS ON NATURAL IMAGE
ax = nexttile(2*n+1,[2 2]);
C = plotmeta.clustMeta(plotmeta.clusts(cc));
ii = 2168; %randi(size(C.fixMeta,1));
thisTrial = C.fixMeta(ii,1);
fixix = find(C.fixMeta(:,1)==thisTrial);
fixix = fixix(1:4);
eyeX = C.fixMeta(fixix,2);
eyeY = C.fixMeta(fixix,3);

try
    Im = imread(fullfile(fileparts(which('marmoV5')), Exp.D{thisTrial}.PR.imagefile));
catch
%     try
%         Im = imread(fullfile(fileparts(which('marmoV5')), strrep(Exp.D{thisTrial}.PR.imageFile, '\', filesep)));
%     catch
end

% zero mean
ppd = Exp.S.pixPerDeg;
Im = mean(Im,3)-127;
Im = imresize(Im, fliplr(Exp.S.screenRect(3:4)));
imagesc(Im); colormap gray
hold on
et = C.fixMeta(fixix,4);
xy = Exp.vpx.smo(et(1):et(end), 2:3)*Exp.S.pixPerDeg.*[1 -1] + Exp.S.centerPix;
% plot(xy(:,1), xy(:,2), 'c')
plot(eyeX - C.rfCenter(1)*ppd, eyeY + C.rfCenter(2)*ppd, 'c');
plot(eyeX - C.rfCenter(1)*ppd, eyeY + C.rfCenter(2)*ppd, 'oc', 'MarkerFaceColor', 'none', 'MarkerSize', 5)
ax.YTick = [];
ax.XTick = [];
tmprect = C.rect + [eyeX(end) eyeY(end) eyeX(end) eyeY(end)];
imrect = [tmprect(1:2) (tmprect(3)-tmprect(1))-1 (tmprect(4)-tmprect(2))-1];
plot([imrect(1) imrect(1) + imrect(3)], imrect([2 2]), 'r', 'Linewidth', 2)
plot([imrect(1) imrect(1) + imrect(3)], imrect(2)+imrect([4 4]), 'r', 'Linewidth', 2)
plot(imrect([1 1]),[imrect(2), imrect(2) + imrect(4)], 'r', 'Linewidth', 2)
plot(imrect(1)+imrect([3 3]), [imrect(2), imrect(2) + imrect(4)], 'r', 'Linewidth', 2)
% xlim([0 size(Im,2)/2])

% -- SPATIAL WINDOW 
ax = nexttile(2*n+3,[1 1]);
I = squeeze(C.fixIms(:,:,fixix(end),2));
imagesc(I);
ax.YTick = []; ax.XTick = [];
axis square
title("ROI")

% -- FFT IN WINDOW
ax = nexttile(3*n+3,[1 1]);
I = abs(squeeze(C.fftIms(:,:,fixix(end),2)));
imagesc(I);
ax.YTick = []; ax.XTick = [];
axis square
title("FFT")

% --- SCHEMATIC RELATIVE RATE
% ax = nexttile(9);
% ax = Axes
%             plot(
%             fixdur(ind)



%%

layout2 = tiledlayout(layout1,1,3);
layout2.Layout.Tile = 3;


%             


%% step through units

cc = 17;



fields = {'rfs_post', 'rfs_pre'};
field = fields{1};
cmap = lines;


figure(100); clf
subplot(3,2,1) % spatial RF
imagesc(Srf{iEx}.xax, Srf{iEx}.yax, Srf{iEx}.spatrf(:,:,cc)); axis xy
hold on
plot(fftrf{iEx}.(field)(cc).rfLocation(1), fftrf{iEx}.(field)(cc).rfLocation(2), 'or')

subplot(3,2,2) % grating fit
imagesc(fftrf{iEx}.(field)(cc).rf.kx, fftrf{iEx}.(field)(cc).rf.ky, fftrf{iEx}.(field)(cc).rf.Ifit')
title(cc)

for  f = 1:2
    field = fields{f};
    subplot(3,2,3+(f-1)*2) % X proj
    bar(fftrf{iEx}.(field)(cc).xproj.bins, fftrf{iEx}.(field)(cc).xproj.cnt, 'FaceColor', .5*[1 1 1]); hold on
    lev = fftrf{iEx}.(field)(cc).xproj.levels(1);
    iix = fftrf{iEx}.(field)(cc).xproj.bins <= lev;
    bar(fftrf{iEx}.(field)(cc).xproj.bins(iix), fftrf{iEx}.(field)(cc).xproj.cnt(iix), 'FaceColor', cmap(5,:));
    lev = fftrf{iEx}.(field)(cc).xproj.levels(2);
    iix = fftrf{iEx}.(field)(cc).xproj.bins >= lev;
    bar(fftrf{iEx}.(field)(cc).xproj.bins(iix), fftrf{iEx}.(field)(cc).xproj.cnt(iix), 'FaceColor', cmap(1,:));
    
    subplot(3,2,4+(f-1)*2) % PSTH
    mrate = fftrf{iEx}.(field)(cc).rateHi;
    srate = fftrf{iEx}.(field)(cc).stdHi / sqrt(fftrf{iEx}.(field)(cc).nHi);
    plot.errorbarFill(fftrf{iEx}.(field)(cc).lags, mrate, srate, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:)); hold on
    mrate = fftrf{iEx}.(field)(cc).rateLow;
    srate = fftrf{iEx}.(field)(cc).stdLow / sqrt(fftrf{iEx}.(field)(cc).nLow);
    plot.errorbarFill(fftrf{iEx}.(field)(cc).lags, mrate, srate, 'k', 'FaceColor', cmap(5,:), 'EdgeColor', cmap(5,:));
    xlim([-.05 .4])
end

plot.suplabel(strrep(sesslist{iEx}, '_', ' '), 't');


% frf = fftrf{iEx}.rfs_post(cc).frf;
% nsteps = numel(fftrf{iEx}.rfs_post(cc).frfsteps);
% figure(10); clf
% clim = [min(frf(:)) max(frf(:))];
% for i = 1:nsteps
%     subplot(1,nsteps+1, i)
%     imagesc(fftrf{iEx}.(field)(cc).rf.kx, fftrf{iEx}.(field)(cc).rf.ky, frf(:,:,i), clim)
%     axis xy
% end


figure(iEx); clf
subplot(321, 'align')
Imap = Srf{iEx}.spatrf(:,:,cc);

xax = Srf{iEx}.xax;
yax = Srf{iEx}.yax;

imagesc(xax, yax, Imap)
colorbar
colormap(plot.viridis)
axis xy
hold on
xlabel('Azimuth (d.v.a)')
ylabel('Elevation (d.v.a)')


% ROI
mu = Srf{iEx}.rffit(cc).mu;
C = Srf{iEx}.rffit(cc).C;

% significance (NEED TO REDO)
if isempty(mu) % no fit was run because RF never crossed threshold
    sig = 0;
else
    ms = (Srf{iEx}.rffit(cc).mushift/Srf{iEx}.rffit(cc).ecc);
    sz = (Srf{iEx}.maxV(cc)./Srf{iEx}.rffit(cc).ecc);
    sig = ms < .25 & sz > 5;
end

zrf = Sgt{iEx}.rf(:,:,cc)*Sgt{iEx}.fs_stim / Sgt{iEx}.sdbase(cc);
z = reshape(zrf(Sgt{iEx}.timeax>=0,:), [], 1);
zthresh = 6;
sigg = sum(z > zthresh) > (1-normcdf(zthresh));

sigs = sig;

fprintf('%d) spat: %d, grat: %d\n', cc, sigs, sigg)


if sigs
    
    offs = trace(C)*10;
    xd = [-1 1]*offs + mu(1);
    xd = min(xd, max(xax)); xd = max(xd, min(xax));
    yd = [-1 1]*offs + mu(2);
    yd = min(yd, max(yax)); yd = max(yd, min(yax));
    
    plot(xd([1 1]), yd, 'r')
    plot(xd([2 2]), yd, 'r')
    plot(xd, yd([1 1]), 'r')
    plot(xd, yd([2 2]), 'r')
    
    plot.plotellipse(mu, C, 1, 'g', 'Linewidth', 2);
    xlabel('Azimuth (d.v.a)')
    ylabel('Elevation (d.v.a)')
end

if sigg && ~isempty(Sgt{iEx}.rffit(cc).srf)
    subplot(3,2,2) % grating RF and fit
    srf = Sgt{iEx}.rffit(cc).srf;
    srf = [rot90(srf,2) srf(:,2:end)];
    
    srfHat = Sgt{iEx}.rffit(cc).srfHat/max(Sgt{iEx}.rffit(cc).srfHat(:));
    srfHat = [rot90(srfHat,2) srfHat(:,2:end)];
    
    xax = [-fliplr(Sgt{iEx}.xax(:)') Sgt{iEx}.xax(2:end)'];
    contourf(xax, -Sgt{iEx}.yax, srf, 10, 'Linestyle', 'none'); hold on
    contour(xax, -Sgt{iEx}.yax, srfHat, [1 .5], 'r', 'Linewidth', 2)
    axis xy
    xlabel('Frequency (cyc/deg)')
    ylabel('Frequency (cyc/deg)')
    title(Sgt{iEx}.rffit(cc).oriPref)
    hc = colorbar;
    
    subplot(3,2,4)
    [~, spmx] = max(reshape(Sgt{iEx}.rffit(cc).srf, [], 1));
    [~, spmn] = min(reshape(Sgt{iEx}.rffit(cc).srf, [], 1));

    cmap = lines;
    plot.errorbarFill(Sgt{iEx}.timeax, Sgt{iEx}.rf(:,spmx,cc)*Sgt{iEx}.fs_stim, Sgt{iEx}.rfsd(:,spmx,cc)*Sgt{iEx}.fs_stim, 'b', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:), 'FaceAlpha', .8); hold on
    plot.errorbarFill(Sgt{iEx}.timeax, Sgt{iEx}.rf(:,spmn,cc)*Sgt{iEx}.fs_stim, Sgt{iEx}.rfsd(:,spmn,cc)*Sgt{iEx}.fs_stim, 'r', 'FaceColor', 'r', 'EdgeColor', 'r', 'FaceAlpha', .8);
    
    ylabel('\Delta Firing Rate (sp s^{-1})')
    xlabel('Time Lag (ms)')
    axis tight

end


% TIME (SPATIAL MAPPING)
subplot(3,2,3)
if ~isnan(Srf{iEx}.spmx(cc))
    plot(Srf{iEx}.timeax, Srf{iEx}.rf(:,Srf{iEx}.spmx(cc),cc)*Srf{iEx}.fs_stim, 'b-o', 'Linewidth', 2); hold on
    plot(Srf{iEx}.timeax, Srf{iEx}.rf(:,Srf{iEx}.spmn(cc),cc)*Srf{iEx}.fs_stim, 'r-o', 'Linewidth', 2);
    plot(Srf{iEx}.peaklagt(cc)*[1 1], ylim, 'k', 'Linewidth', 2)
    plot(Sgt{iEx}.peaklagt(cc)*[1 1], ylim, 'k--', 'Linewidth', 2)
    
    xlabel('Time Lag (ms)')
    ylabel('Firing Rate')
end

if sigg && Sgt{iEx}.peaklag(cc) > 0
% PLOTTING FIT
[xx,yy] = meshgrid(Sgt{iEx}.rffit(cc).oriPref/180*pi, 0:.1:15);
X = [xx(:) yy(:)];





% plot data RF tuning
par = Sgt{iEx}.rffit(cc).pHat;

lag = Sgt{iEx}.peaklag(cc);

fs = Sgt{iEx}.fs_stim;
srf = reshape(Sgt{iEx}.rf(lag,:,cc)*fs, Sgt{iEx}.dim);
srfeb = reshape(Sgt{iEx}.rfsd(lag,:,cc)*fs, Sgt{iEx}.dim);

if Sgt{iEx}.ishartley
    
    [kx,ky] = meshgrid(Sgt{iEx}.xax, Sgt{iEx}.yax);
    sf = hypot(kx(:), ky(:));
    sfs = min(sf):max(sf);
    
    ori0 = Sgt{iEx}.rffit(cc).oriPref/180*pi;
    
    [Yq, Xq] = pol2cart(ori0*ones(size(sfs)), sfs);
    
    r = interp2(Sgt{iEx}.xax, -Sgt{iEx}.yax, srf, Xq, Yq);
    eb = interp2(Sgt{iEx}.xax, -Sgt{iEx}.yax, srfeb, Xq, Yq);
    
    subplot(3,2,6)
    h = errorbar(sfs, r, eb, 'ok'); hold on
    h.CapSize = 0;
    h.MarkerSize = 2;
    h.MarkerFaceColor = 'k';
    h.LineWidth = 1.5;
    xlabel('Spatial Frequency (cpd)')
    xlim([0 8])
    
    oris = 0:(pi/10):pi;
    sf0 = Sgt{iEx}.rffit(cc).sfPref;
    [Yq, Xq] = pol2cart(oris, sf0*ones(size(oris)));
    
    r = interp2(Sgt{iEx}.xax, -Sgt{iEx}.yax, srf, Xq, Yq, 'linear');
    eb = interp2(Sgt{iEx}.xax, -Sgt{iEx}.yax, srfeb, Xq, Yq);
    
    subplot(3,2,5)
    h = errorbar(oris/pi*180, r, eb, 'ok'); hold on
    h.CapSize = 0;
    h.MarkerSize = 2;
    h.MarkerFaceColor = 'k';
    h.LineWidth = 1.5;
    xlabel('Orientation (deg)')
    ylabel('Firing Rate')
    xlim([0 180])
    
else
    %%
    [i,j] = find(srf == max(srf(:)));
    ori0 = Sgt{iEx}.xax(j)/180*pi;
    sf0 = Sgt{iEx}.yax(i);
    
    subplot(3,2,5)
    errorbar(Sgt{iEx}.xax, srf(i,:), srfeb(i,:), 'o-', 'Linewidth', 2); hold on
    xlim([0 180])
    xlabel('Orientation (deg)')
    ylabel('Firing Rate')
    
    subplot(3,2,6)
    errorbar(Sgt{iEx}.yax, srf(:,j), srfeb(:,j), 'o-', 'Linewidth', 2); hold on
    xlabel('Spatial Frequency (cpd)')
    
end

% if sigg % Only plot fits if it's "significant"
    orientation = 0:.1:pi;
    spatfreq = 0:.1:20;
    
    % plot tuning curves
    orientationTuning = prf.parametric_rf(par, [orientation(:) ones(numel(orientation), 1)*sf0], strcmp(Sgt{iEx}.sftuning, 'loggauss'));
    spatialFrequencyTuning = prf.parametric_rf(par, [ones(numel(spatfreq), 1)*ori0 spatfreq(:)], strcmp(Sgt{iEx}.sftuning, 'loggauss'));
    
    subplot(3,2,5)
    plot(orientation/pi*180, orientationTuning, 'r', 'Linewidth', 2)
    
    
    subplot(3,2,6)
    plot(spatfreq, spatialFrequencyTuning, 'r', 'Linewidth', 2)
%     plot(par(3)*[1 1], ylim, 'b')
    
    
    % plot bandwidth
    if strcmp(Sgt{iEx}.sftuning, 'loggauss')
        a = sqrt(-log(.5) * par(4)^2 * 2);
        b1 = (par(3) + 1) * exp(-a) - 1;
        b2 = (par(3) + 1) * exp(a) - 1;
    else
        a = acos(.5);
        logbase = log(par(4));
        b1 = par(3)*exp(-a*logbase);
        b2 = par(3)*exp(a*logbase);
    end
    
%     plot(b2*[1 1], ylim, 'b')
%     plot(b1*[1 1], ylim, 'k')
%     plot( Sgt{iEx}.rffit(cc).sfPref*[1 1], ylim)
end


plot.suplabel(sprintf('%s: %d', strrep(sesslist{iEx}, '_', ' '), cc), 't');
plot.fixfigure(gcf, 10, [4 8], 'offsetAxes', false)