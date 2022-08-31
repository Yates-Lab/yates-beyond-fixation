%% add paths
user = 'jakelaptop';
addFreeViewingPaths(user);
addpath Analysis/manuscript_freeviewingmethods/
%% ROIs for fovea sessions
% We have to export the stimulus for offline calibration which uses pytorch
% This code will reconstruct the stimulus at full resolution for relevant
% stimulus paradigms and copy the resulting hdf5 file to the server for
% analysis

% everything reconstructed within an ROI: This is set
RoiS = struct();
RoiS.('logan_20200304') = [-20 -60 50 10];
RoiS.('logan_20200306') = [-20 -60 50 10];
RoiS.('logan_20191231') = [-20 -60 50 10];
RoiS.('logan_20191119') = [-20 -60 50 10];
RoiS.('logan_20191120') = [-20 -50 50 20];
RoiS.('logan_20191120a') = [-20 -50 50 20];
RoiS.('logan_20191121') = [-20 -50 50 20];
RoiS.('logan_20191122') = [-20 -50 50 20];


%%

sesslist = io.dataFactory();

%% load session
close all
sessId = sesslist{56}; %'logan_20200304';
spike_sorting = 'kilowf';
[Exp, S] = io.dataFactory(sessId, 'spike_sorting', spike_sorting, 'cleanup_spikes', 0);
S.rect([2 4]) = sort(-S.rect([2 4]));

eyePosOrig = Exp.vpx.smo(:,2:3);

% correct eye-position offline using the eye position measurements on the
% calibration trials if they were saved
% try
    eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);
    
    lam = .5; % mixing between original eye pos and corrected eye pos
    Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);
% end


%% get visually driven units
[spkS, W] = io.get_visual_units(Exp, 'plotit', false);

%% Load STA analysis from Python

fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'Dots', 'Gabor', 'BackImage'}, 'overwrite', false);


%%
sid = regexp(sessId, '_', 'split');

fname2 = ['Figures/2021_pytorchmodeling/rfs_' sid{2} '_' spike_sorting '.mat'];
if exist(fname2, 'file')
    tmp = load(fname2);
end


% tmp.shiftx = tmp.shiftx / 60 * 5.2;
% tmp.shifty = tmp.shifty / 60 * 5.2;
%% plot shifter

figure(2); clf
subplot(1,2,1)
contourf(tmp.xspace, tmp.yspace, tmp.shiftx*60/5.2, 'w'); colorbar
subplot(1,2,2)
contourf(tmp.xspace, tmp.yspace, tmp.shifty*60/5.2, 'w'); colorbar


%%
amp = zeros(NC, 2);
for cc = 1:NC
    sta = tmp.stas_pre(:,:,:,cc);
    sd = std(sta(:));
    sta2 = tmp.stas_post(:,:,:,cc);
    
    sta = sta./sd;
    sta2 = sta2./sd;
    amp(cc,1) = max(sta(:));
    amp(cc,2) = max(sta2(:));
end

figure(1); clf
plot(amp(:,1), amp(:,2), 'o')
hold on
xd = xlim;
plot(xd, xd, 'k')
plot(xd, 1.5*xd, 'k--')

figure(2); clf
plot(abs(tmp.mushiftx(:)), tmp.sdshiftx(:), 'o'); hold on
plot(abs(tmp.mushifty(:)), tmp.sdshifty(:), 'o');
%%
sta = tmp.stas_post(:,:,:,1);
    NX = size(sta,3);
    NY = size(sta,2);
NC = size(tmp.stas_post,4);
cmap = [[ones(128,1) repmat(linspace(0,1,128)', 1, 2)]; [flipud(repmat(linspace(0,1,128)', 1, 2)) ones(128,1)]];
xax = linspace(-1,1,NX)*30;
yax = linspace(-1,1,NY)*30;

figure(1); clf
for cc = 1:NC
%     sta = stas(:,:,:,W(cc).cid);
    sta = tmp.stas_post(:,:,:,cc);
    
    nlags = size(sta,1);
    sta = reshape(sta, nlags, []);
    sta = (sta - mean(sta(:))) ./ std(sta(:));
    [bestlag, ~] = find(max(abs(sta(:)))==abs(sta));
    extrema = max(max(sta(:)), max(abs(sta(:))));
    I = reshape(sta(bestlag,:), [NX NY])';
    ind = 10:60;
    I = I(ind, ind);
    imagesc(xax(ind) + W(cc).x + cc*30, yax(ind) + W(cc).depth, I, 1*[-1 1]*extrema); hold on
    xlim([-50 30*NC])
    ylim([-50 1.1*max([W.depth])])
    drawnow
    
end

colormap(cmap)

%%
cc = cc + 1;
NC = numel(W);
if cc > NC
    cc = 1;
end


sta = tmp.stas_post(:,:,:,cc);
NX = size(sta,3);
NY = size(sta,2);
nlags = size(sta,1);
sta = reshape(sta, nlags, []);
sta = (sta - mean(sta(:))) ./ std(sta(:));
[bestlag, ~] = find(max(abs(sta(:)))==abs(sta));


figure(1); clf

subplot(2,3,1)
plot(W(cc).wavelags, W(cc).ctrChWaveform, 'k'); hold on
plot(W(cc).wavelags, W(cc).ctrChWaveformCiHi, 'k--')
plot(W(cc).wavelags, W(cc).ctrChWaveformCiLo, 'k--')
title(cc)
axis tight

subplot(2,3,4);
plot(W(cc).lags, W(cc).isi)
title(W(cc).isiV)
xlim([0 20])

extrema = max(max(sta(:)), max(abs(sta(:))));

subplot(2,3,3)
imagesc(reshape(sta(bestlag,:), [NX NY])', [-1 1]*extrema)
colormap(cmap)

[~, imx] = max(sta(bestlag,:));
[~, imn] = min(sta(bestlag,:));

subplot(2,3,6)
plot(sta(:,imx), 'b'); hold on
plot(sta(:,imn), 'r')
% 
% iEx = find(strcmp(sessId,sesslist));
% if Sgt{iEx}.sig(cc) && ~isempty(Sgt{iEx}.rffit(cc).srf)
%     subplot(2,3,2)
% %     ax = subplot(3,1,2); % grating RF and fit
%     srf = Sgt{iEx}.rffit(cc).srf;
%     srf = [rot90(srf,2) srf(:,2:end)];
%     
%     srfHat = Sgt{iEx}.rffit(cc).srfHat/max(Sgt{iEx}.rffit(cc).srfHat(:));
%     srfHat = [rot90(srfHat,2) srfHat(:,2:end)];
%     
%     xax = [-fliplr(Sgt{iEx}.xax(:)') Sgt{iEx}.xax(2:end)'];
%     contourf(xax, -Sgt{iEx}.yax, srf, 10, 'Linestyle', 'none'); hold on
%     contour(xax, -Sgt{iEx}.yax, srfHat, [1 .5], 'r', 'Linewidth', 1)
%     axis xy
%     xlabel('Frequency (cyc/deg)')
%     ylabel('Frequency (cyc/deg)')
% %     title(Sgt{iEx}.rffit(cc).oriPref)
%     hc = colorbar;
%     
%     subplot(2,3,5)
%     [~, spmx] = max(reshape(Sgt{iEx}.rffit(cc).srf, [], 1));
%     [~, spmn] = min(reshape(Sgt{iEx}.rffit(cc).srf, [], 1));
% 
%     clrs = [0 0 0; .5 .5 .5];
%     plot.errorbarFill(Sgt{iEx}.timeax, Sgt{iEx}.rf(:,spmx,cc)*Sgt{iEx}.fs_stim, Sgt{iEx}.rfsd(:,spmx,cc)*Sgt{iEx}.fs_stim, 'b', 'FaceColor', clrs(1,:), 'EdgeColor', clrs(1,:), 'FaceAlpha', .8); hold on
%     plot.errorbarFill(Sgt{iEx}.timeax, Sgt{iEx}.rf(:,spmn,cc)*Sgt{iEx}.fs_stim, Sgt{iEx}.rfsd(:,spmn,cc)*Sgt{iEx}.fs_stim, 'r', 'FaceColor', clrs(2,:), 'EdgeColor', clrs(2,:), 'FaceAlpha', .8);
%     
%     ylabel('\Delta Firing Rate (sp s^{-1})')
%     xlabel('Time Lag (ms)')
%     axis tight
% 
% end
%%
figure(3); clf
xd = 1:NX;
yd = 1:NY;

dt = 20;
[xx,yy] = meshgrid(xd, yd);
xoffsets = linspace(0, 200, nlags);
yoffsets = linspace(20, 0, nlags)*0;
h = [];

for ilag = 1:nlags
    I = reshape(sta(ilag,:), [NX NY])';
    h = surf(xx+xoffsets(ilag), zeros(size(xx))+ilag*dt, yy+yoffsets(ilag), I); hold on
end

shading flat
colormap(sqrt(cmap))
axis equal
view(50, 15)
% 
% colormap gray
shading flat
caxis([-1 1]*extrema)
axis off



%% If you want to recompute in matlab using shifter
stim = 'Gabor';
tset = 'Train';

% Stimulus
Stim = h5read(fname, ['/' stim '/' tset '/Stim']);
StimS = zeros(size(Stim), 'like', Stim);

% Frame Times
ftoe = h5read(fname, ['/' stim '/' tset '/frameTimesOe']);
frate = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'frate');

% Spikes
st = h5read(fname, ['/Neurons/' spike_sorting '/times']);
clu = h5read(fname, ['/Neurons/' spike_sorting '/cluster']);
cids = h5read(fname, ['/Neurons/' spike_sorting '/cids']);
sp.st = st;
sp.clu = clu;
sp.cids = cids;
Robs = binNeuronSpikeTimesFast(sp, ftoe, 1/frate);

% Eye position at frame (for shifting and inclusion criterion)
eyeAtFrame = h5read(fname, ['/' stim '/' tset '/eyeAtFrame']);
labels = h5read(fname, ['/' stim '/' tset '/labels']);
valinds = h5read(fname, ['/' stim '/' tset '/valinds']);

NX = size(Stim,2);
NY = size(Stim,3);

eyeX = (eyeAtFrame(:,2) - Exp.S.centerPix(1)) / Exp.S.pixPerDeg;
eyeY = (eyeAtFrame(:,3) - Exp.S.centerPix(2)) / Exp.S.pixPerDeg;

figure(1); clf
subplot(2,1,1)
plot(eyeX); hold on
plot(eyeY)
title("Eye At Frame")

shiftX = interp2(tmp.xspace, tmp.yspace, tmp.shiftx, eyeX, eyeY, 'linear', 0);
shiftY = interp2(tmp.xspace, tmp.yspace, tmp.shifty, eyeX, eyeY, 'linear', 0);

% %% test
% eyeX = -0.4549;
% eyeY = 2.6173;
% shiftX = interp2(tmp.xspace, tmp.yspace, tmp.shiftx, eyeX, eyeY, 'linear', 0);
% shiftY = interp2(tmp.xspace, tmp.yspace, tmp.shifty, eyeX, eyeY, 'linear', 0);
% 
% [shiftX shiftY]
% %%
% shiftX = single(.5);
% shiftY = single(.5);
% subplot(2,1,2)
% plot(shiftX); hold on
% plot(shiftY)
% title("Shift")
% 
% % Shift stimulus
dims = size(Stim);
NT = dims(1);
dims(1) = [];
% 
xax = linspace(-1,1,dims(2));
yax = linspace(-1,1,dims(1));
[xgrid,ygrid] = meshgrid(xax(1:dims(2)), yax(1:dims(1)));


%%
figure(1); clf
set(gcf, 'Color', 'w')
iFrame = 100%iFrame + 1;
xsample = xgrid+0;%shiftX(iFrame);
ysample = ygrid+0;%shiftY(iFrame);
I = interp2(xgrid, ygrid, single(squeeze(Stim(iFrame,:,:)))', xsample, ysample, 'linear', 0);
StimS(iFrame,:,:) = I;

I0 = single(squeeze(Stim(iFrame,:,:)));
figure(1); clf
subplot(1,3,1)
imagesc(I0', [-1 1]*127); colormap gray
axis off
subplot(1,3,2)
imagesc(I, [-1 1]*127); axis off
subplot(1,3,3)
imagesc(I0-I)
drawnow
%%


disp("shift correcting stimulus...")
for iFrame = 1:NT
    xsample = xgrid+shiftX(iFrame);
    ysample = ygrid+shiftY(iFrame);
    I = interp2(xgrid, ygrid, single(squeeze(Stim(iFrame,:,:)))', xsample, ysample, 'linear', 0);
    StimS(iFrame,:,:) = I;
    
%     I0 = single(squeeze(Stim(iFrame,:,:)));
%     figure(1); clf
%     subplot(1,3,1)
%     imagesc(I0)
%     subplot(1,3,2)
%     imagesc(I)
%     subplot(1,3,3)
%     imagesc(I0-I)
%     drawnow
end
disp("Done")

%%


% compute STA with forward correlation
NC = size(Robs,2);
nlags = 15;
Rdelta = Robs - mean(Robs);
nstim = size(Stim,2);
ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc < 5.2 & labels == 1;
ix = sum(conv2(double(ix), eye(nlags), 'full'),2) == nlags; % all timelags good

stas = zeros(nlags, dims(1), dims(2), NC);
for ii = 1:dims(1)
    fprintf('%d/%d\n', ii, dims(1))
    for jj = 1:dims(2)
        Xstim = conv2(StimS(:,ii,jj), eye(nlags), 'full');
        Xstim = Xstim(1:end-nlags+1,:);
        stas(:, ii,jj, :) = Xstim(ix,:)'*Rdelta(ix,:);
    end
end

%% plot all STAs as a function of depth

NC = numel(W);

cmap = [[ones(128,1) repmat(linspace(0,1,128)', 1, 2)]; [flipud(repmat(linspace(0,1,128)', 1, 2)) ones(128,1)]];

c1 = [0.1216    0.4667    0.7059];

% c2 = [0.1725    0.6275    0.1725];
c2 = [0.8392    0.1529    0.1569]
cmap = [ [linspace(c1(1), .9, 128); linspace(c1(2), .9, 128); linspace(c1(3), .9, 128)]'; ...
    flipud([linspace(c2(1), .9, 128); linspace(c2(2), .9, 128); linspace(c2(3), .9, 128)]')];

%     
xax = linspace(-1,1,NX)*30;
yax = linspace(-1,1,NY)*30;

figure(1); clf
for cc = 1:NC
    sta = stas(:,:,:,W(cc).cid);
%     sta = tmp.stas_post(:,:,:,cc);
    nlags = size(sta,1);
    sta = reshape(sta, nlags, []);
    sta = (sta - mean(sta(:))) ./ std(sta(:));
    [bestlag, ~] = find(max(abs(sta(:)))==abs(sta));
    extrema = max(max(sta(:)), max(abs(sta(:))));
    I = reshape(sta(bestlag,:), [NX NY])';
    ind = 10:60;
    I = I(ind, ind);
%     abs(I).^10
    imagesc(xax(ind) + W(cc).x + cc*30, yax(ind) + W(cc).depth, I, .5*[-1 1]*extrema); hold on
    xlim([-50 NC*30])
    ylim([-50 1.1*max([W.depth])])
    drawnow
    
end

colormap(cmap)

%%
figure(2); clf
cc = cc + 1;
NC = numel(W);
if cc > NC
    cc = 1;
end


sta = stas(:,:,:,W(cc).cid);



% nlags = size(tmp.stas_pre,1);

nlags = size(sta,1);
sta = reshape(sta, nlags, []);
sta = (sta - mean(sta(:))) ./ std(sta(:));
[bestlag, ~] = find(max(abs(sta(:)))==abs(sta));
% [~, bestlag] = max(max(sta,[],2));
figure(1); clf

subplot(2,3,1)
plot(W(cc).wavelags, W(cc).ctrChWaveform, 'k'); hold on
plot(W(cc).wavelags, W(cc).ctrChWaveformCiHi, 'k--')
plot(W(cc).wavelags, W(cc).ctrChWaveformCiLo, 'k--')
title(cc)
axis tight
ax = subplot(2,2,3);
plot(W(cc).lags, W(cc).isi)
title(W(cc).isiV)
% ax.XScale ='log';

extrema = max(max(sta(:)), max(abs(sta(:))));
subplot(2,3,2)
imagesc(reshape(sta(bestlag,:), [NX NY])', [-1 1]*extrema)

% sta2 = stas(:,:,:,W(cc).cid);
sta2 = tmp.stas_post(:,:,:,cc);
sta2 = (sta2 - mean(sta2(:))) / std(sta2(:));
[bestlag, ~] = find(max(abs(sta2(:)))==abs(sta2));
subplot(2,3,3)
imagesc(squeeze(sta2(bestlag,:,:))',  [-1 1]*extrema)

% colormap(plot.viridis)
colormap(cmap)

% axis xy

[~, imx] = max(sta(bestlag,:));
[~, imn] = min(sta(bestlag,:));

subplot(2,2,4)
plot(sta(:,imx), 'b'); hold on
plot(sta(:,imn), 'r')


figure(3); clf
for ilag = 1:nlags
    subplot(2,ceil(nlags/2),ilag)
    imagesc(reshape(sta(ilag,:), [NX NY]), [-1 1]*extrema)
end
colormap(sqrt(cmap))


%%
figure(4); clf

h = [];
for ilag = 1:nlags
    I = reshape(sta(ilag,:), [NX NY])';
    h = surf(xx, zeros(size(xx))+ilag*dt, yy, I); hold on
end

shading flat
colormap(sqrt(cmap))
axis equal
view(0, -5)
% 
% colormap gray
shading flat
caxis([-1 1]*extrema)
%%

xd = 1:NX;
yd = 1:NY;

dt = 20;
[xx,yy] = meshgrid(xd, yd);
xoffsets = linspace(0, 200, nlags);
yoffsets = linspace(0, 200, nlags);
h = [];

for ilag = 1:nlags
    h = surf(xx+xoffsets(ilag), zeros(size(xx))+ilag*dt, yy+yoffsets(ilag), squeeze(sta(ilag,:,:))); hold on
end

% alphas = linspace(1, .25, nframes);
% 
% h = [];
% f = [];

% for iframe = 1:nframes
%     h = surf(xx+xoffsets(iframe), zeros(size(xx))+iframe*dt, yy+yoffsets(iframe), Stim(:,:,iframe)); hold on
%     h.FaceAlpha = alphas(iframe);
% %     fill3(xd([1 1 end end])+xoffsets(iframe), iframe*dt*[1 1 1 1], yd([1 end end 1])+yoffsets(iframe), 'k'); hold on %, 'FaceColor', 'None', 'EdgeColor', 'k')
% end
% 
axis equal
view(0, -5)
% 
% colormap gray
shading flat
caxis([-1 1])


%%
sta = squeeze(tmp.stas_post(:,:,:,cc));



%%
[xx,tt,yy] = meshgrid(xax, (1:nlags)*8, yax);
% I = permute(sta, [3 1 2]);
I = sta;

figure(2); clf
set(gcf, 'Color', 'w')
h = slice(xx,tt, yy, I, [], (1:10)*8,[]);
set(gca, 'CLim', [-9 9])
for i = 1:numel(h)
    h(i).EdgeColor = 'none';
%     text(-12, i*8-4, 5, sprintf('%i ms', (i)*8))
end
view(79,11)
colormap(cmap)
axis off

hold on
plot3(10+[10 16], [1 1]*39, -[50 50], 'r', 'Linewidth', 2)

plot.fixfigure(gcf, 8, [14 3])
% saveas(gcf, fullfile('Figures', 'K99', sprintf('sta%02.0f.png', cc)))