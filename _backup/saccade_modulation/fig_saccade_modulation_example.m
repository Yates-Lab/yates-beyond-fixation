

user = 'jakelaptop';
addFreeViewingPaths(user);
addpath scripts/
addpath saccade_modulation/
figDir = 'figures/fig02';

datadir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'preprocessed');
sesslist = arrayfun(@(x) x.name, dir(fullfile(datadir, '*.mat')), 'uni', 0);
%% Load session

isess = 1;
Exp = load(fullfile(datadir, sesslist{isess}));
Exp.FileTag = sesslist{isess};

%% get spatial RF and grating RF

srf = get_spat_rf_coarsefine_reg(Exp);
grf = grat_rf_basis_reg(Exp, 'plot', false, 'debug', false);

%% get saccade modulation
[rf_post, plotmeta] = fixrate_by_fftrf(Exp, srf, grf, 'debug', false, 'plot', false, 'usestim', 'post', 'alignto', 'fixon', 'binsize', 10e-3);

%% plot it
         
fig = figure(3); clf
ax = gca;

cc = 2;


fixdur = plotmeta.fixdur;
[~, ind] = sort(fixdur);
[i,j] = find(squeeze(plotmeta.spks(ind,cc,:)));


% -- RASTER
f1 = find(fixdur(ind) > .20,1);
f2 = find(fixdur(ind) > .50,1);
ix = i >= f1 & i <= f2;
lags = rf_post(1).lags;

plot.raster(lags(j(ix)), i(ix), 5); axis tight
hold on
cmap = lines;
plot(fixdur(ind(f1:f2)), f1:f2, '-', 'Color', cmap(4,:))

xlabel('Time from fixation (s)')
ylabel('Fixation #')
ax.YTick = [];
ax.Box = 'off';
ax.TickDir = 'out';
ax.TickLength = ax.TickLength*2;

%%
% -- FIXATIONS ON NATURAL IMAGE
figure(2); clf
ax = subplot(1,2,1);

C = plotmeta.clustMeta(plotmeta.clusts(cc));
ii = 2168; %randi(size(C.fixMeta,1));
thisTrial = C.fixMeta(ii,1);
fixix = find(C.fixMeta(:,1)==thisTrial);
fixix = fixix(1:4);
eyeX = C.fixMeta(fixix,2);
eyeY = C.fixMeta(fixix,3);

Im = imread(fullfile(fileparts(which('marmoV5')), Exp.D{thisTrial}.PR.imagefile));

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


% -- SPATIAL WINDOW 
ax = subplot(2,2,2);
I = squeeze(C.fixIms(:,:,fixix(end),2));
imagesc(I);
ax.YTick = []; ax.XTick = [];
axis square
title("ROI")

% -- FFT IN WINDOW
ax = subplot(2,2,4);
I = abs(squeeze(C.fftIms(:,:,fixix(end),2)));
imagesc(I);
ax.YTick = []; ax.XTick = [];
axis square
title("FFT")
      

%% step through units

cc = 11;

fields = {'rfs_post'};
field = fields{1};
cmap = lines;


figure(100); clf
subplot(3,2,1) % spatial RF
imagesc(srf.RFs(cc).xax, srf.RFs(cc).yax, srf.RFs(cc).srf); axis xy
hold on
plot(rf_post(cc).rfLocation(1), rf_post(cc).rfLocation(2), 'or')

subplot(3,2,2) % grating fit
imagesc(rf_post(cc).rf.kx, rf_post(cc).rf.ky, rf_post(cc).rf.Ifit')
title(cc)

for  f = 1
    field = fields{f};
    subplot(3,2,3+(f-1)*2) % X proj
    bar(rf_post(cc).xproj.bins, rf_post(cc).xproj.cnt, 'FaceColor', .5*[1 1 1]); hold on
    lev = rf_post(cc).xproj.levels(1);
    iix = rf_post(cc).xproj.bins <= lev;
    bar(rf_post(cc).xproj.bins(iix), rf_post(cc).xproj.cnt(iix), 'FaceColor', cmap(5,:));
    lev = rf_post(cc).xproj.levels(2);
    iix = rf_post(cc).xproj.bins >= lev;
    bar(rf_post(cc).xproj.bins(iix), rf_post(cc).xproj.cnt(iix), 'FaceColor', cmap(1,:));
    
    subplot(3,2,4+(f-1)*2) % PSTH
    mrate = rf_post(cc).rateHi;
    srate = rf_post(cc).stdHi / sqrt(rf_post(cc).nHi);
    plot.errorbarFill(rf_post(cc).lags, mrate, srate, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:)); hold on
    mrate = rf_post(cc).rateLow;
    srate = rf_post(cc).stdLow / sqrt(rf_post(cc).nLow);
    plot.errorbarFill(rf_post(cc).lags, mrate, srate, 'k', 'FaceColor', cmap(5,:), 'EdgeColor', cmap(5,:));
    xlim([-.05 .4])
end