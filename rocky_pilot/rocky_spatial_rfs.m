% Exp=load('/home/huklab/Documents/SpikeSorting/Output/2023-06-08/r20230608_dpi_redo')
Exp = rocky_datafactory(2);
eyePos=Exp.vpx.smo(:,2:3);



%% Get spatial map
dotTrials = io.getValidTrials(Exp, 'Dots');
if ~isempty(dotTrials)
    
    BIGROI = [-10 -10 10 10];
    %eyePos = C.refine_calibration();
    binSize = .2;
    Frate = 60;
    [Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
        'ROI', BIGROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
        'eyePosExclusion', 2e3, ...
        'eyePos', eyePos, 'frate', Frate, 'fastBinning', true);
    
    % use indices while fixating
    ecc = hypot(opts.eyePosAtFrame(:,1), opts.eyePosAtFrame(:,2))/Exp.S.pixPerDeg;
    ix = opts.eyeLabel==1 & ecc < 15.2;
    
end


spike_rate = mean(RobsSpace)*Frate;
%%


figure(2); clf; set(gcf, 'Color', 'w')
subplot(3,1,1:2)
h = [];
h(1) = stem(spike_rate, '-ok', 'MarkerFaceColor', 'k', 'MarkerSize', 4);

ylabel('Firing Rate During Stimulus')
hold on
goodunits = find(spike_rate > .1);
h(2) = plot(goodunits, spike_rate(goodunits), 'og', 'MarkerFaceColor', 'g', 'MarkerSize', 4);
h(3) = plot(xlim, [1 1], 'r-');
legend(h, {'All units', 'Good Units', '1 Spike/Sec'}, 'Location', 'Best')

subplot(3,1,3)
stem(spike_rate, '-ok', 'MarkerFaceColor', 'k', 'MarkerSize', 4)
ylim([0, 1])
xlabel('Unit #')
ylabel('Firing rate of bad units')


fprintf('%d / %d fire enough spikes to analyze\n', numel(goodunits), size(RobsSpace,2))
drawnow
%
win = [-3 15];
stas = forwardCorrelation(Xstim, mean(RobsSpace,2), win);
stas = stas / std(stas(:)) - mean(stas(:));
wm = [min(stas(:)) max(stas(:))];
nlags = size(stas,1);
figure(1); clf
for ilag = 1:nlags
    subplot(2, ceil(nlags/2), ilag)
    imagesc(opts.xax, opts.yax, reshape(stas(ilag, :), opts.dims), wm)
end

%% Analyze population
R = RobsSpace(:,goodunits);
%stas = forwardCorrelation(Xstim, R-mean(R), win, find(ix));
stas = forwardCorrelation(Xstim, R-mean(R), win);
NC = size(R,2);

%% plotting
sx = 5;
sy = 5; %ceil(NC/sx);
nperfig = sx*sy;
nfigs = ceil(NC / nperfig);
fprintf('%d figs\n', nfigs)

if nfigs == 1
    figure(1); clf; set(gcf, 'Color', 'w')
end

for cc = 1:NC
    if nfigs > 1
        fig = ceil(cc/nperfig);
        figure(fig); set(gcf, 'Color', 'w')
    end

    si = mod(cc, nperfig);  
    if si==0
        si = nperfig;
    end
    subplot(sy,sx, si)

    I = squeeze(stas(:,:,cc));
    I = (I - mean(I(:)) )/ std(I(:));

    [bestlag,j] = find(I==max(I(:)));
    I = I(min(max(bestlag(1)+[-1 0 1], 1), nlags),:);
    I = mean(I);
%     I = std(I);
    xax = opts.xax/Exp.S.pixPerDeg;
    yax = opts.yax/Exp.S.pixPerDeg;
    imagesc(xax, yax, imgaussfilt(reshape(I, opts.dims), 1), [-2, 2])
    colormap(plot.coolwarm)
    title(Exp.osp.cids(goodunits(cc)))
end

%% find subset of good units

bestlag = zeros(NC,1);
spatrf = zeros(prod(opts.dims), NC);

figure(1); clf
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
ax = plot.tight_subplot(sx, sy, 0.001, 0, 0);

for cc = 1:NC
    I = squeeze(stas(:,:,cc));
    I = (I - mean(I(:)) )/ std(I(:));

    [bestlag(cc),j] = find(I==max(I(:)), 1);
    
    I = I(min(max(bestlag(cc)+[-1 0 1], 1), nlags),:);
    I = imgaussfilt(reshape(mean(I), opts.dims), 1);
    spatrf(:,cc) = I(:);
%     set(gcf, 'currentaxes', ax(cc))
%     imagesc(I)
end

%%

ss = zeros(NC, 1);
for i = 1:NC
    [u,s,v] = svd(reshape(spatrf(:,i), opts.dims));
    ss(i) = s(1)./sum(diag(s));
end

gm = fitgmdist(ss, 2);
[idx,NLOGL,POST,LOGPDF,MAHALAD] = gm.cluster(ss);

figure(1); clf
for i = unique(idx(:))'
    plot(ss(i==idx), '.'); hold on
end

inds = find(POST(:,2)> 0.05);
clf
plot(ss, '.'); hold on
plot(inds, ss(inds), '.')
% inds = find(idx==find(gm.mu==max(gm.mu)));

%% plot subset

subset = [407 408 411 419 473 477 507 520 570 571 577 580 581 585 609 611 ...
    656 657 677 695 700 708 709 714 722 725 735 748 775 799 842 844 850 889 890 891 902 908 911  ...
    933 934 944 945 946 950 955 968 975 1002 1010 1021 1022 1023 1029 1031 1041 1048 1077 1100 1152 1181 1184 1189 1192 ...
    1202 1219 1239 1252 1253 1274 1277 1284 1286 1292 1293 1307 1308 1314 1318 1350 1364 1399 1402 1414 1419 1451 1462 ...
    1467 1485 1487 1500 1505 1506 1513 1521 1655 1661 1668 1671 1672 1725 1736 1958 1959 1970 1982];

% subset = Exp.osp.cids(goodunits(inds))';
%%
nsub = numel(subset);
[~, ccid] = ismember(subset, Exp.osp.cids(goodunits));
%[~, ccid] = ismember(subset, Exp.osp.cids);

nfigs = ceil(nsub / nperfig);
fig = ceil(1); clf
figure(fig); set(gcf, 'Color', 'w')

sx = ceil(sqrt(nsub));
sy = round(sqrt(nsub));

for cc = 1:nsub
    subset(cc)
%     if nfigs > 1
%         fig = ceil(cc/nperfig);
%         figure(fig); set(gcf, 'Color', 'w')
%     end
% 
%     si = mod(cc, nperfig);  
%     if si==0
%         si = nperfig;
%     end
si=cc;
    subplot(sy,sx, si)

    I = squeeze(stas(:,:,ccid(cc)));
    I = (I - mean(I(:)) )/ std(I(:));

    [bestlag,j] = find(I==max(I(:)));
    I = I(min(max(bestlag(1)+[-1 0 1], 1), nlags),:);
    I = mean(I);
%     I = std(I);
    xax = opts.xax/Exp.S.pixPerDeg;
    yax = opts.yax/Exp.S.pixPerDeg;
    imagesc(xax, yax, imgaussfilt(reshape(I, opts.dims), 1), [-2, 2])
    colormap(plot.coolwarm)
    title(Exp.osp.cids(goodunits(ccid(cc))))
    title(Exp.osp.clusterDepths(goodunits(ccid(cc))))
    grid on
end