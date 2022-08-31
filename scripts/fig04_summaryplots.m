fid = 1;
figstyle= 'nature';

%% loop over sessions and load RF analyses

flist = dir('Figures/2021_pytorchmodeling/rfs*.mat');

figDir = 'Figures/manuscript_freeviewing/fig04';

% 1) shifter patterns consistent
% 2) shift amounts
% 3) before / after examples (spatial)
% 4) spatiotemporal examples
% 5) before / after amplitude

numSessions = numel(flist);

rfs = repmat(struct('stas', [], ...
    'rfpre', [], ...
    'rfpost', [], ...
    'amppre', [], ...
    'amppost', [], ...
    'sig', [], ...
    'shiftxmu', [], ...
    'shiftymu', [], ...
    'shiftxsd', [], ...
    'shiftysd', [], ...
    'maxV', []), numSessions, 1);
    
zthresh = 5;  % rfs are zscored and thresholded to find significant units
plotit = false;

for ex = 1:numSessions
    
    if ex==1
        extent = [-10 60 -60 10];
    else
        extent = [-20 50 -50 20];
    end
    
    % load analyses for this session
    tmp = load(fullfile(flist(ex).folder, flist(ex).name));
    
    NC = size(tmp.stas_post,4);
    NX = size(tmp.stas_post,2);
    NY = size(tmp.stas_post,3);
    
    yax = linspace(extent(3),extent(4),NY);
    xax = linspace(extent(1),extent(2),NX);
    
    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));
    
    % mod2 = mod(sx,2);
    % sx = sx + mod2;
    % sy = sy - mod2;
    
    if plotit
        figure(1); clf
        figure(2); clf
    end
    
    rfs(ex).cids = tmp.cids;
    
    % shifter consistentcy
    rfs(ex).shiftxmu = tmp.mushiftx;
    rfs(ex).shiftymu = tmp.mushifty;
    rfs(ex).shiftxsd = abs(tmp.sdshiftx);
    rfs(ex).shiftysd = abs(tmp.sdshifty);
    
    % RFs and summary statistics
    nlags = size(tmp.stas_pre,1);
    rfs(ex).stas = zeros(nlags, NY, NX, NC);
    rfs(ex).rfpre = zeros(NY, NX, NC);
    rfs(ex).rfpost = zeros(NY, NX, NC);
    rfs(ex).amppre = zeros(NC, 1);
    rfs(ex).amppost = zeros(NC, 1);
    rfs(ex).maxV = zeros(NC,1);
    rfs(ex).sig = zeros(NC,1);
    rfs(ex).xax = xax;
    rfs(ex).yax = yax;
    
    
    for cc = 1:NC
        
        sta = tmp.stas_pre(:,:,:,cc);
        sta2 = tmp.stas_post(:,:,:,cc);
        
        % zscore
        sta = (sta - mean(sta(:))) ./ std(sta(:));
        sta2 = (sta2 - mean(sta2(:))) ./ std(sta2(:));
        
        for ilag = 1:nlags
            rfs(ex).stas(ilag,:,:,cc) = squeeze(sta2(ilag,:,:))';
        end
        
        % find regions above threshold
        s = regionprops3(bwlabeln(abs(sta2)>zthresh));
        
        if isempty(s)
            if plotit
                figure(1)
                subplot(sx, sy, cc)
                plot(abs(sta2(:)))
            end
        else
            if plotit
                figure(1)
                subplot(sx, sy, cc)
                plot(abs(sta2(:)), 'r')
                title(max(s.Volume))
            end
            
            rfs(ex).maxV(cc) = max(s.Volume);
            
        end
        
        nlags = size(sta,1);
        sflat = reshape(sta2, nlags, []);
        
        [bestlag, ~] = find(max(abs(sflat(:)))==abs(sflat));
        
        extrema = max(max(sta2(:)), max(abs(sta2(:))));
        
        I1 = squeeze(sta(bestlag,:,:))';
        I2 = squeeze(sta2(bestlag,:,:))';
        
        rfs(ex).rfpre(:,:,cc) = I1;
        rfs(ex).rfpost(:,:,cc) = I2;
        
        if plotit
            figure(2)
            subplot(sx, sy, cc)
            imagesc(xax, yax, I2, [-1 1]*extrema); hold on
            axis xy
        end
        
        rfs(ex).amppre(cc) = abs(max(I1(:))-min(I1(:)));
        rfs(ex).amppost(cc) = abs(max(I2(:))-min(I2(:)));
        
    end
    
    uscore = strfind(flist(ex).name, '_');
    sessid = flist(ex).name(uscore(1)+1:uscore(2)-1);
    Exp = io.dataFactory(['logan_' sessid], 'spike_sorting', 'kilowf');
    
    [spkS, Wf] = io.get_visual_units(Exp);
    
    rfs(ex).spkS = spkS;
    rfs(ex).Wf = Wf;
    
    rfs(ex).sessid = sessid;
    
    if plotit
        colormap(plot.coolwarm) %#ok<*UNRCH>
    end
end

%% plot shifter

ex = 6;
xax = linspace(-5,5,100);

t = tiledlayout(2,1);
t.TileSpacing = 'compact';
t.Padding = 'compact';

nexttile
contourf(xax, xax, rfs(ex).shiftxmu, 20, 'Linestyle', 'none')
axis square
colorbar
set(gca, 'XTick', -5:2.5:5, 'YTick', -5:2.5:5)
grid on

nexttile
contourf(xax, xax, rfs(ex).shiftymu, 20, 'Linestyle', 'none')
axis square
colorbar
colormap(plot.viridis)
set(gca, 'XTick', -5:2.5:5, 'YTick', -5:2.5:5)
grid on

% t.XLabel.String = "Gaze Position (d.v.a.)";
% t.YLabel.String = "Gaze Position (d.v.a.)";

plot.formatFig(gcf, [1.3 1.4], figstyle);
saveas(gcf,sprintf('Figures/manuscript_freeviewing/fig04/shifter_%s.pdf', rfs(ex).sessid))
%% shifter max / median values and consistency across runs
rng(1234) % fix jitter

% max values
mxXmu = arrayfun(@(x) max(abs(x.shiftxmu(:))), rfs(:));
mxYmu = arrayfun(@(x) max(abs(x.shiftymu(:))), rfs(:));
mxXsd = arrayfun(@(x) max(x.shiftxsd(:)), rfs(:));
mxYsd = arrayfun(@(x) max(x.shiftysd(:)), rfs(:));
% median 
mdXmu = arrayfun(@(x) median(abs(x.shiftxmu(:))), rfs(:));
mdYmu = arrayfun(@(x) median(abs(x.shiftymu(:))), rfs(:));
mdXsd = arrayfun(@(x) median(x.shiftxsd(:)), rfs(:));
mdYsd = arrayfun(@(x) median(x.shiftysd(:)), rfs(:));

%%

figure(1); clf
jitter = randn(numSessions,1)*.1;
cmap = lines;
cmap2 = (.3*ones(size(cmap)) + .7*cmap);

XTicks = [.5 1 2.5 3];
h = errorbar(XTicks(1)*ones(numSessions,1)+jitter, mxXmu, mxXsd, 'o', 'Color', cmap(3,:), 'MarkerFaceColor', cmap(3,:), 'CapSize', 0, 'MarkerSize', 2.5); hold on
h(2) = errorbar(XTicks(2)*ones(numSessions,1)+jitter, mxYmu, mxYsd, 'o', 'Color', cmap2(3,:), 'MarkerFaceColor', cmap2(3,:), 'CapSize', 0, 'MarkerSize', 2.5);

h = errorbar(XTicks(3)*ones(numSessions,1)+jitter, mdXmu, mdXsd, 'o', 'Color', cmap(4,:), 'MarkerFaceColor', cmap(4,:), 'CapSize', 0, 'MarkerSize', 2.5); hold on
h(2) = errorbar(XTicks(4)*ones(numSessions,1)+jitter, mdYmu, mdYsd, 'o', 'Color', cmap2(4,:), 'MarkerFaceColor', cmap2(4,:), 'CapSize', 0, 'MarkerSize', 2.5);

set(gca, 'XTick', XTicks, 'XTickLabel', {'Horizontal (max)', 'Vertical (max)', 'Horizontal (median)', 'Vertical (median)'}, 'XTickLabelRotation', -45)
ylabel('Shift Amount (arcmin)')

xlim([0 4])
yd = ylim;
ylim([0 yd(2)])

plot.formatFig(gcf, [2 2], figstyle)
saveas(gcf, fullfile(figDir, 'shiftamount.pdf'))



%% example before after
exs = [1, 1,  2, 3, 3,   3,  4,  5, 5, 6,  6, 6];
ccs = [71, 73,8,  11, 62, 64, 21, 3, 7, 23, 28, 57];

% %%
% ex = 5;
% cc = 0;
% %%

for i = 1:numel(exs)
    
    ex = exs(i);

    cc = ccs(i);

%     %%
%     cc = cc + 1;
% if cc > size(rfs(ex).rfpre,3)
%     cc = 1;
% end
    figure(i); clf
    t = tiledlayout(1,2);
    t.TileSpacing = 'Compact';
    
    pre = rfs(ex).rfpre(:,:,cc);
    post = rfs(ex).rfpost(:,:,cc);
    extrema = max(abs(post(:)))*1.25;
    
    nexttile
    imagesc(xax, yax, pre, extrema*[-1 1])
    axis xy
%     title('Before', 'Fontweight', 'normal')
    grid on
    
    nexttile
    imagesc(xax, yax, post, extrema*[-1 1])
    set(gca, 'YTickLabel', [])
    axis xy
    hold on
    grid on
%     title(cc)
    
%     title('After', 'Fontweight', 'normal')
    
    colormap(plot.coolwarm)
    h = colorbar;
%     xlabel(t, 'Space (arcminutes)')
%     ylabel(t, 'Space (arcminutes)')

% %%
    exname = strrep(flist(ex).name, '_kilowf.mat', '');
    plot.formatFig(gcf, [2 1], figstyle)
    saveas(gcf, fullfile(figDir, sprintf('%s_%d.pdf', exname, cc)))
    
end


%% 3D spatiotemporal RF
 
for i = 1:numel(exs)
    ex = exs(i);
    cc = ccs(i);

    figure(i); clf

    sta = rfs(ex).stas(:,:,:,cc);
    corners = plot.sliceRF(sta, 'dt', 50);
    view(50, 10)
    
%     plot3(corners([1 end],1)+50, corners([1 end],2), corners([1 end],3)-10, 'k')
    
    for ilag = 1:size(corners,1)
        x = corners(ilag,1)+50;
        y = corners(ilag,2);
        z = corners(ilag,3) - 10;
        if ismember(ilag, [1 5 12])
            plot3(x+[0 0], y+ [0 0], z + [0 -5], 'k')
            text(x, y, z-20, sprintf('%d', ilag*8 + 8))
        end
        
    end
    
    plot3(corners(5,1)+40+[0 10*1.6], corners(5,2)*[1 1], corners(1,3)*[1 1], 'k', 'Linewidth', 2)
    
    exname = strrep(flist(ex).name, '_kilowf.mat', '');
    plot.formatFig(gcf, [2 1], figstyle)
%     set(gcf, 'Color', 'None')
%     set(gca, 'Color', 'None')

    saveas(gcf, fullfile(figDir, sprintf('spatiotemporal%s3D_%d.pdf', exname, cc)))
end


%% again with less fancy plotting
for i = 1:numel(exs)
    ex = exs(i);
    cc = ccs(i);

    figure(i); clf
    
    
    sta = rfs(ex).stas(:,:,:,cc);
    
    numLags = size(sta,1);
    t = tiledlayout(1, numLags);
    t.TileSpacing = 'compact';
    
    extrema = max(max(sta(:)), max(-sta(:)));

    for ilag = 1:numLags
        nexttile
        imagesc(rfs(ex).xax, rfs(ex).yax, squeeze(sta(ilag,:,:)), [-1 1]*extrema);
        set(gca, 'XTick', rfs(ex).xax(1):10:rfs(ex).xax(end), ...
            'YTick', rfs(ex).yax(1):10:rfs(ex).yax(end), 'XTickLabel', '', 'YTickLabel', '')
        grid on
        axis xy
        axis square
        hold on

    end
    
    colormap(plot.coolwarm)
    
    exname = strrep(flist(ex).name, '_kilowf.mat', '');
    plot.formatFig(gcf, [numLags 1], figstyle)

    saveas(gcf, fullfile(figDir, sprintf('spatiotemporalLags%s_%d.pdf', exname, cc)))
end
%% Begin some analyses

cmap = lines;

nEx = numel(rfs);
rfdist = zeros(nEx,1);
rfecc = zeros(nEx,2);

figure(3); clf
figure(4); clf

% Loop over experiments, compute RF centers, eccentricity, contours
for ex = 1:nEx
    
    NC = size(rfs(ex).rfpost,3);
    ctr = nan(NC,2);
    ar = nan(NC,1);
    ecc = nan(NC,1);
    wfx = nan(NC,1);
    com = nan(NC,2);
    contours = cell(NC,1);
    
    figure(1); clf
    
    for cc = 1:NC
        
        post = rfs(ex).rfpost(:,:,cc);
        dim = size(post);
        
        extrema = max(abs(post(:)))*1.25;
        
        [xi, yi] = meshgrid((rfs(ex).xax), rfs(ex).yax);
        
        sda = abs(post);
        sda = imgaussfilt(sda, 2);
        
        [xc, yc] = radialcenter(sda.^4);
        xc = interp1(1:dim(1), (rfs(ex).xax), xc);
        yc = interp1(1:dim(1), rfs(ex).yax, yc);
        
        com(cc,1) = xc;
        com(cc,2) = yc;
        
        [xy, ar(cc), ctr(cc,:)] = get_rf_contour(xi, yi, sda, 'plot', true, 'thresh', 3);
        ecc(cc) = hypot(ctr(cc,1), ctr(cc,2));
        wfx(cc) = rfs(ex).Wf(cc).x;
        
        figure(2); clf
        subplot(1,2,1)
        imagesc(xi(:), yi(:), post); hold on
        plot(com(cc,1), com(cc,2), 'or', 'MarkerFaceColor', 'r', 'Linewidth', 2)
        
        subplot(1,2,2)
        imagesc(xi(:), yi(:), sda); hold on
        plot(com(cc,1), com(cc,2), 'or')
        pause(.1)
        
        contours{cc} = xy;
        
    end
    
    rfs(ex).ctrs = ctr;
    rfs(ex).area = ar;
    rfs(ex).ecc = ecc;
    rfs(ex).wfx = wfx;
    rfs(ex).com = com;
    rfs(ex).contours = contours;
    
    
    wfamp=[rfs(ex).Wf.peakval]'-[rfs(ex).Wf.troughval]';
    wftim = arrayfun(@(x) x.peaktime, rfs(ex).Wf) - arrayfun(@(x) x.troughtime, rfs(ex).Wf);
    
    good = wftim*1e3 > 0 & wftim*1e3<.5;
    good = good & wfamp > 40;
    good = good & cellfun(@(x) size(x,1)>1, contours);
    good = good & rfs(ex).amppost(:) > 10;
    [th, rho] = cart2pol(com(:,1), com(:,2));
    
    figure(4);
    ix1 = abs(wfx-0) < 1e-3 & good;
    polarplot(th(ix1), rho(ix1), 'o', 'Color', cmap(ex,:), 'MarkerSize', 2, 'MarkerFaceColor', cmap(ex,:)); hold on
    ix2 = abs(wfx-200) < 1e-3 & good;
    clr = cmap(ex,:)*.5 + [1 1 1]*.5;
    polarplot(th(ix2), rho(ix2), 'o', 'Color', clr, 'MarkerSize', 2, 'MarkerFaceColor', clr);
    
    mu = mean([com(ix1,1), com(ix1,2)]);
    C = cov(com(ix1,:));
    plot.plotellipsepolar(mu, C, 1, 'b', 'Color', cmap(ex,:))
    
    mu = mean([com(ix2,1), com(ix2,2)]);
    C = cov(com(ix2,:));
    plot.plotellipsepolar(mu, C, 1, 'r', 'Color', clr)
    
    figure(3);
    plot(com(ix1,1), com(ix1,2), '.b'); hold on
    plot(com(ix2,1), com(ix2,2), '.r'); hold on
    
    
    mu1 = mean([com(ix1,1), com(ix1,2)]);
    C = cov(com(ix1,:));
    plot.plotellipse(mu1, C, 1, 'b')
    
    mu2 = mean([com(ix2,1), com(ix2,2)]);
    C = cov(com(ix2,:));
    plot.plotellipse(mu2, C, 1, 'r')
    
    xlim([-1 1]*30)
    ylim([-1 1]*30)
    grid on
    
    rfdist(ex) = sqrt(sum((mu1-mu2).^2));
    rfecc(ex,1) = hypot(mu1(1),mu1(2));
    rfecc(ex,2) = hypot(mu2(1),mu2(2));
end

plot.formatFig(4, [4 4], 'poster')
saveas(gcf, 'Figures/2021_ucbtalk/rfcenters.pdf')

%% rf size / position
figure(1); clf
   

enum = cell2mat(arrayfun(@(x) find(strcmp(x.sessid, {rfs.sessid}))*ones(numel(x.ecc), 1), rfs, 'uni', 0));
ccs = cell2mat(arrayfun(@(x) (1:numel(x.ecc))', rfs, 'uni', 0));
ecc = cell2mat(arrayfun(@(x) x.ecc(:), rfs, 'uni', 0));
amp = cell2mat(arrayfun(@(x) x.amppost(:), rfs, 'uni', 0));
ar = cell2mat(arrayfun(@(x) x.area(:), rfs, 'uni', 0));
csize = cell2mat(arrayfun(@(x) cellfun(@(y) isnan(y(1)), x.contours), rfs, 'uni', 0));
wfamp = cell2mat(arrayfun(@(x) [x.Wf.peakval]'-[x.Wf.troughval]', rfs, 'uni', 0));
wftim = cell2mat(arrayfun(@(x) [x.Wf.peaktime]'-[x.Wf.troughtime]', rfs, 'uni', 0));

good = wftim*1e3 > 0 & wftim*1e3<.5;
good = good & wfamp > 40;
good = good & amp > 13;

lm = fitlm(ecc(good), sqrt(ar(good)));
lm.plot;
hold on
sz = sqrt(ar);
plot(ecc(good), sz(good), 'ok', 'MarkerFaceColor', .5*[1 1 1])
ylim([0 20])
xlabel('Eccentricity (arcmin)')
ylabel('RF Size (sqrt(Area))')
title('')
plot.formatFig(1, [4 4], 'default')
saveas(gcf, 'Figures/2021_ucbtalk/rfsize.pdf')

plot(xlim, m*xlim+7)

%% 
figure(2); clf
amppre = cell2mat(arrayfun(@(x) x.amppre, rfs, 'uni', 0));
amppost = cell2mat(arrayfun(@(x) x.amppost, rfs, 'uni', 0));

amppre = amppre(good);
amppost = amppost(good);

clr = cmap(1,:);
plot(amppre, amppost, 'o', 'Color', .8*[1 1 1], 'MarkerFaceColor', clr, 'MarkerSize', 2, 'Linewidth', .25); hold on
xd = [0 ceil(max(amppost(:)))];
xd = [0 35];
xlim(xd)
ylim(xd)
plot(xd, xd, 'k')
md = median(amppost./amppre);

mdci = bootci(1000, @median, amppost./amppre);
fprintf(fid, 'Median amp ratio is: %02.3f [%02.3f, %02.3f]\n', md, mdci(1), mdci(2))

[pval, ~, stats] = signrank(amppost,amppre);
fprintf('signrank test: p = %d (%02.5f), %02.3f, %d\n', pval, pval, stats.zval, stats.signedrank)

plot(xd, md*xd, 'k:')
% fill([xd fliplr(xd)], [mdci(1)*xd mdci(2)*fliplr(xd)], 'k', 'FaceAlpha', .2, 'EdgeColor', 'none')
title('RF amplitude')
xlabel('Before (z)')
ylabel('After (z)')

plot.formatFig(gcf, [1 1], figstyle)
saveas(gcf, fullfile(figDir, 'ampcompare.pdf'))

plot.formatFig(gcf, [5 5], 'default')
saveas(gcf, fullfile(figDir, 'ampcompareTalk.pdf'))

%% plot RF distance at 200µm in arcminutes

figure(1); clf
rfe = min(rfecc,[],2);

lm = fitlm(rfe,rfdist);
lm.plot;
title('')
hold on
plot(rfe, rfdist, 'ok', 'MarkerFaceColor', .5*[1 1 1])
xlabel('Eccentricity (arcmin)')
ylabel('RF Distance (arcmin)')
plot.formatFig(1, [4 4], 'default')
saveas(gcf, 'Figures/2021_ucbtalk/rfdistance.pdf')

m = lm.Coefficients.Estimate(2);


%%
md = median(sz(good));
mdci = bootci(1e3, @median, sz(good));
n = sum(good);
fprintf('Median RF size = %02.2f [%02.2f,%02.2f] arcminutes (n=%d)\n', md, mdci, n)

md = median(ecc(good));
mdci = bootci(1e3, @median, ecc(good));
n = sum(good);
fprintf('Median ecc = %02.2f [%02.2f,%02.2f] arcminutes (n=%d)\n', md, mdci, n)


%%
id = find(ecc < 16 & sz > 14);
figure(1); clf
imagesc(rfs(enum(id)).xax, rfs(enum(id)).yax, rfs(enum(id)).rfpost(:,:,ccs(id))); hold on
plot(rfs(enum(id)).contours{ccs(id)}(:,1), rfs(enum(id)).contours{ccs(id)}(:,2), 'k')


%%
id = find(enum==6 & ccs == 58);
plot(ecc(id), sz(id), 'or')
plot.formatFig(1, [4 4], 'default')
saveas(gcf, 'Figures/2021_ucbtalk/rfsize2.pdf')

%%
disp(lm)

figure(2); clf
cmap = lines;
for i = 1:numel(rfs)
    ix = i==enum & good;
    plot(ecc(ix), sqrt(ar(ix)), 'ok', 'MarkerFaceColor', cmap(i,:)); hold on
end
legend({rfs.sessid})
    
%% example contours / centers


figure(2); clf
for i = 1:numel(exs)
    ex = exs(i);
    cc = ccs(i);
    post = rfs(ex).rfpost(:,:,cc);
    dim = size(post);
    
    extrema = max(abs(post(:)))*1.25;
    
    [xi, yi] = meshgrid((rfs(ex).xax), rfs(ex).yax);
    
    figure(1); clf
    imagesc(xi(:), yi(:), post, [-1 1]*extrema); hold on
    title(cc)
    
    sda = abs(post);
    sda = imgaussfilt(sda, 2);
    
    [xc, yc] = radialcenter(sda.^4);
    xc = interp1(1:dim(1), (rfs(ex).xax), xc);
    yc = interp1(1:dim(1), rfs(ex).yax, yc);
    
    [xy, ar(cc), ctr(cc,:)] = get_rf_contour(xi, yi, sda, 'plot', false, 'thresh', 3);
    plot(xy(:,1), xy(:,2), 'k', 'Linewidth', 1.5)
    xlabel('Space (arcmin)')
    ylabel('Space (arcmin)')
    set(gcf, 'Color', 'w')
    plot(xc, yc, '+k', 'Linewidth', 1.5)
    axis xy
    
    plot.formatFig(gcf, [4 4], 'default', 'pretty', true)
    
    
    title('')
    saveas(gcf, sprintf('Figures/2021_ucbtalk/example%d_%d.pdf', ex, cc))
end
%% RFs across the probes
for ex = 1:6
    NC = size(rfs(ex).rfpost,3);
    NX = size(rfs(ex).rfpost,2);
    NY = size(rfs(ex).rfpost,1);
    
    cmap = plot.coolwarm;
    xax = linspace(-1,1,NX)*60;
    yax = linspace(-1,1,NY)*60;
    
    cmap2 = lines;
    cmap3 = lines;
    cmap3(1,:) = [];
    % figure(1); clf
    % for i = 1:6
    %     plot(i,i, 'o', 'Color', cmap2(i,:), 'MarkerFaceColor', cmap2(i,:)); hold on
    %     plot(i,i+.5, 'o', 'Color', cmap3(i,:), 'MarkerFaceColor', cmap3(i,:)); hold on
    % end
    % %%
    figure(1); clf
    figure(2); clf
    [~, cinds] = sort(rfs(ex).wfx);
    for i = 1:NC
        cc = cinds(i);
        
        if rfs(ex).amppost(cc) < 10 || size(rfs(ex).contours{cc},1)==1
            continue
        end
        
        post = rfs(ex).rfpost(:,:,cc);
        dim = size(post);
        
        extrema = max(abs(post(:)))*1.25;
        
        I = post;
        
        x = rfs(ex).contours{cc}(:,1);
        y = rfs(ex).contours{cc}(:,2);
        figure(2);
        if rfs(ex).Wf(cc).x < 100
            clr = cmap2(1,:)*.75 + [1 1 1]*.25;
        else
            clr = cmap2(5,:)*.5 + [1 1 1]*.5;
        end
        fill(x, y, 'r', 'FaceColor', clr, 'Linewidth', 1.5, 'FaceAlpha', .25, 'EdgeColor', 'none'); hold on
        plot(x,y,'Color', clr)
        figure(1);
        x = interp1(rfs(ex).xax, xax, x);
        y = interp1(rfs(ex).yax, yax, y);
        
        ind = 1:NX;
        I = I(ind, ind);
        xoff = rfs(ex).Wf(cc).x + cc*30;
        yoff = rfs(ex).Wf(cc).depth;
        imagesc(xax(ind) + xoff, yax(ind) + yoff, I, 1*[-1 1]*extrema); hold on
        plot(x + xoff, y  + yoff, 'k', 'Linewidth', 2);
        xlim([-50 35*NC])
        ylim([-50 1.1*max([rfs(ex).Wf.depth])])
        drawnow
        
    end
    
    colormap(cmap)
    
    plot.formatFig(1, [8 4], 'default', 'pretty', false)
    
    title('')
    saveas(1, sprintf('Figures/2021_ucbtalk/rfdepths%d.pdf', ex))
    
    figure(2)
    xlim([-1 1]*40)
    ylim([-1 1]*40)
    plot.polar_grid(0:30:360, 0:10:40, 'TextColor', [1 1 1])
    
    plot.formatFig(2, [12 12], 'poster', 'pretty', false)
    saveas(2, sprintf('Figures/2021_ucbtalk/rfcontours%d.pdf', ex))
end
%% coordinates of cropping for NIM
cropidxs = struct();
cropidxs.('l20191119_kilowf') = [21,50,1,30];
cropidxs.('l20191120a_kilowf') = [21,50,21,50];
cropidxs.('l20191121_kilowf') = [21,50,21,50];
cropidxs.('l20191122_kilowf') = [11,50,21,50];
cropidxs.('l20191202_kilowf') = [11,50,16,45];
cropidxs.('l20191205_kilowf') = [11,50,16,45];
cropidxs.('l20191206_kilowf') = [11,50,16,45];
cropidxs.('l20200304_kilowf') = [11,40,16,50];

%%
ex = 6;
fname = sprintf('Figures/2021_pytorchmodeling/%s_kilowf_model2.mat', rfs(ex).sessid);
if exist(fname, 'file')
    fprintf('Loading [%s]\n', fname)
    tmp = load(fname);
end

crop = cropidxs.(['l' rfs(ex).sessid '_kilowf']);

%% plot subunits 
wsub = tmp.wsubunits(:,:,:,:);

isub = isub + 1;

if isub > size(wsub,1)
    isub = 1;
end
sub = squeeze(wsub(isub,:,:,:));
sub = (sub - mean(sub(:))) ./ std(sub(:));

sd = squeeze(std(sub));
[i,j] = find(sd == max(sd(:)));

[~, bestlag] = max(abs(sub(:,i,j)));
extrema = max(abs(sub(:)));

figure(1); clf
imagesc(squeeze(sub(bestlag,:,:)))
axis off
title(isub)
colormap gray
%% threshold subunits by readout weight
thresh = 0;
nsubs = size(tmp.wreadout,2);
ninh = ceil(.1*nsubs);
nexc = nsubs - ninh;

wfamp=[rfs(ex).Wf.peakval]'-[rfs(ex).Wf.troughval]';
wftim = arrayfun(@(x) x.peaktime, rfs(ex).Wf) - arrayfun(@(x) x.troughtime, rfs(ex).Wf);
    

probeix = {abs(rfs(ex).wfx) < 1e-3, abs(rfs(ex).wfx-200) < 1e-3};
good = wftim*1e3 > 0 & wftim*1e3<.5;
good = good & wfamp > 40;
% % good = good(:) & tmp.ll0(:)>.2;

nshank = numel(probeix);
figure(1); clf
for i = 1:nshank
    ix = good(:) & probeix{i}(:);
%     subplot(1,nshank,i)
    plot(sum(tmp.wreadout(ix,:))); hold on
end
plot(sum(tmp.wreadout), 'k'); hold on
plot(xlim, [1 1]*thresh, 'k--')

%%
thresh = 0;
figure(1); clf
nsubs = size(tmp.wsubunits,1);
goodsubs = find(sum(tmp.wreadout(tmp.ll0>.2,1:nsubs))>=thresh);
goodsubs = 1:nsubs;
sd = squeeze(std(tmp.wsubunits,[],2));
nsubs = size(tmp.wsubunits,1);
mxsd = zeros(nsubs, 1);
for i = 1:nsubs
    sub = squeeze(tmp.wsubunits(i,:,:,:));
    sub = (sub - mean(sub(:))) ./ std(sub(:));
    
    sd = squeeze(std(sub));
    
    mxsd(i) = max(sd(:));
end

% goodsubs= intersect(goodsubs, find(mxsd>2));


% % wsub = tmp.wsubunits(goodsubs,:,:,:);
% nsubs = numel(goodsubs);
% 
% amps = zeros(nsubs,1);
% for isub = 1:nsubs
%     sub = squeeze(wsub(isub,:,:,:));
%     sub = (sub - mean(sub(:))) ./ std(sub(:));
%     amps(isub) = max(sub(:)) - min(sub(:));
% end
% 
% goodsubs = find(amps > 0);
wreadout = tmp.wreadout(:,goodsubs);
wsub = tmp.wsubunits(goodsubs,:,:,:);
nsubs = numel(goodsubs);

% figure(1); clf
% plot(amps)
% %%

wdth = 1/nsubs;
steps = linspace(0, 1-wdth*2, nsubs);
ax = zeros(nsubs,1);
for isub = 1:nsubs
    
    if goodsubs(isub)<=nexc
        ax(isub) = axes('Position', [steps(isub), .5+mod(isub,2)*wdth*2.5, wdth*1.8, wdth*1.8]);
    else
        ax(isub) = axes('Position', [steps(isub), .7+mod(isub,2)*wdth*2.5, wdth*1.8, wdth*1.8]);
    end
    
    axis off
    
    sub = squeeze(wsub(isub,:,:,:));
    sub = (sub - mean(sub(:))) ./ std(sub(:));
    
    sd = squeeze(std(sub));
    [i,j] = find(sd == max(sd(:)));
    
    [~, bestlag] = max(abs(sub(:,i,j)));
    extrema = max(abs(sub(:)));
    imagesc(squeeze(sub(bestlag,:,:)), [-1 1]*max(0, extrema))
%     sd = squeeze(std(sub));
%     imagesc(sd)
    axis off
    title([isub max(sd(:))])
   
    
end
colormap(gray.^1.2)

axes('Position', [0.05 0.1 .9 .2])
imagesc(wreadout')
% ix = good(:) & probeix{1}(:);
% plot(wreadout(ix,:)', 'bo-', 'MarkerSize', 2); hold on
% ix = good(:) & probeix{2}(:);
% plot(wreadout(ix,:)', 'o-r', 'MarkerSize', 2); hold on
% xlim([1 nsubs])

%% loop over subunits and lasso
sz = size(wsub);
NY = sz(end-1);
NX = sz(end);

xax = rfs(ex).xax(crop(1):crop(2));
yax = rfs(ex).yax(crop(3):crop(4));

[xi,yi] = meshgrid( xax, yax);

% isub = 1;%isub + 1;
ss = repmat(struct('rf', [], 'timeon', [], 'timeoff', [], ...
    'contouron', [], 'contouroff', []), nsubs,1);

for isub = 1:nsubs
    
    % if isub > nsubs
    %     isub = 1;
    % end
    sub = squeeze(wsub(isub,:,:,:));
    sub = (sub - mean(sub(:))) ./ std(sub(:));
    
    sd = squeeze(std(sub));
    [i,j] = find(sd == max(sd(:)));
    
    [~, bestlag] = max(abs(sub(:,i,j)));
    
    figure(2); clf
    subplot(1,2,1)
    extrema = max(abs(sub(:)));
    I = squeeze(sub(bestlag,:,:));
    [imx,jmx] = find(I == max(I(:)));
    [imn,jmn] = find(I == min(I(:)));
    imagesc(xi(:), yi(:), I, [-1 1]*max(10, extrema)); hold on
    colormap(gray)
    
    [con, aron] = get_rf_contour(xi, yi, I, 'thresh', 4, 'plot', false);
    [coff, aroff] = get_rf_contour(xi, yi, -I, 'thresh', 4, 'plot', false);
    [csub, arsub] = get_rf_contour(xi, yi, imgaussfilt(abs(I),1), 'thresh', 3, 'plot', false);
    
    plot(con(:,1), con(:,2), 'r')
    plot(coff(:,1), coff(:,2), 'b')
    plot(csub(:,1), csub(:,2), 'k')
    plot(10 + [0 5], [1 1]*-25, 'k', 'Linewidth', 2)
    axis xy
    
    
    
    subplot(1,2,2)
    plot(sub(:,imx,jmx), 'r'); hold on
    plot(sub(:,imn,jmn), 'b')
    
    % get subunit centers
    r = max(imgaussfilt(I, 1), 0).^4;
    [cx,cy] = radialcenter(r);
    cx = interp1(1:numel(xax), xax, cx);
    cy = interp1(1:numel(yax), yax, cy);
    conxy = [cx cy];
    
    r = max(imgaussfilt(-I, 1), 0).^4;
    [cx,cy] = radialcenter(r);
    cx = interp1(1:numel(xax), xax, cx);
    cy = interp1(1:numel(yax), yax, cy);
    coffxy = [cx cy];
    
    r = imgaussfilt(abs(I), 1).^4;
    [cx,cy] = radialcenter(r);
    cx = interp1(1:numel(xax), xax, cx);
    cy = interp1(1:numel(yax), yax, cy);
    csubxy = [cx cy];
    
    ss(isub).rf = I;
    ss(isub).xax = xax;
    ss(isub).yax = yax;
    ss(isub).timeon = sub(:,imx,jmx);
    ss(isub).timeoff = sub(:,imn,jmn);
    ss(isub).contoursub = csub;
    ss(isub).contouron = con;
    ss(isub).contouroff = coff;
    ss(isub).arsub = arsub;
    ss(isub).aron = aron;
    ss(isub).aroff = aroff;
    ss(isub).ctrsubxy = csubxy;
    ss(isub).ctroffxy = coffxy;
    ss(isub).ctronxy = conxy;
    
    title(isub)
%     pause(1)
    plot.fixfigure(gcf, 10, [4 2], 'offsetAxes', false)
    saveas(gcf, sprintf('Figures/2021_ucbtalk/subunit_%d_%d.pdf', ex, isub))
end

ampthresh = 11;
ampon = arrayfun(@(x) max(x.rf(:)), ss);
ampoff = arrayfun(@(x) -min(x.rf(:)), ss);
amps = ampon + ampoff;
arsub = arrayfun(@(x) x.arsub, ss(amps>ampthresh));
aron = arrayfun(@(x) x.aron, ss(amps>ampthresh & ampon > 4));
aroff = arrayfun(@(x) x.aroff, ss(amps>ampthresh & ampoff > 4));

eccsub = arrayfun(@(x) hypot(x.ctrsubxy(1), x.ctrsubxy(2)), ss(amps>11));

eccon = arrayfun(@(x) hypot(x.ctronxy(1), x.ctronxy(2)), ss(amps>ampthresh & ampon > 4));
eccoff = arrayfun(@(x) hypot(x.ctroffxy(1), x.ctroffxy(2)), ss(amps>ampthresh & ampoff > 4));


% plot centers
figure(1); clf
for isub = 1:nsubs
    if amps(isub) < 10
        continue
    end
    if ampon(isub) > 4
        fill(ss(isub).contouron(:,1), ss(isub).contouron(:,2), 'r', 'FaceColor', 'r', 'FaceAlpha', .25); hold on
    end
    if ampoff(isub)>4
        fill(ss(isub).contouroff(:,1), ss(isub).contouroff(:,2), 'b', 'FaceColor', 'b', 'FaceAlpha', .25);
    end
end

xlim(rfs(ex).xax([1 end]))
ylim(rfs(ex).yax([1 end]))




figure(20);
plot(eccsub, sqrt(arsub), 'o'); hold on
% plot(eccon, sqrt(aron), 'o'); hold on
% plot(eccoff, sqrt(aroff), 'o')

%%
% NC = size(tmp.stas,4);
% for cc = 1:NC
cc = cc + 1;
if cc > NC
    cc = 1;
end
figure(1); clf
subplot(1,2,1)
sub = tmp.stas(:,:,:,cc);
sub = (sub - mean(sub(:))) ./ std(sub(:));
    
    sd = squeeze(std(sub));
    [i,j] = find(sd == max(sd(:)));
    
    [~, bestlag] = max(abs(sub(:,i,j)));
    
    subplot(1,2,1)
    extrema = max(abs(sub(:)));
    I = squeeze(sub(bestlag,:,:));
    [imx,jmx] = find(I == max(I(:)));
    [imn,jmn] = find(I == min(I(:)));
imagesc(rfs(ex).xax, rfs(ex).yax, I', [-1 1]*extrema); hold on
plot(rfs(ex).contours{cc}(:,1), rfs(ex).contours{cc}(:,2), 'k', 'Linewidth', 1.5)
axis xy
sub = tmp.stasHat(:,:,:,cc);
sub = (sub - mean(sub(:))) ./ std(sub(:));
I = squeeze(sub(bestlag,:,:));
extrema = max(abs(sub(:)));
subplot(1,2,2)
imagesc(rfs(ex).xax, rfs(ex).yax, I', [-1 1]*extrema); hold on
plot(rfs(ex).contours{cc}(:,1), rfs(ex).contours{cc}(:,2), 'k', 'Linewidth', 1.5)
axis xy
title(cc)
grid on
set(gca, 'GridColor', 'r')
%%

plot.fixfigure(gcf, 10, [5 2], 'offsetAxes', false)
saveas(gcf, sprintf('Figures/2021_ucbtalk/stamodel_%d_%d.pdf', ex, cc))
figure(2); clf
plot(tmp.wreadout(cc,:), '-o')

%%
yax = rfs(ex).yax((crop(1):crop(2))+1);
xax = rfs(ex).xax((crop(3):crop(4))+1);

goodsubs = find(tmp.wreadout(cc,:)>.1);
wreadout = tmp.wreadout(:,goodsubs);
wsub = tmp.wsubunits(goodsubs,:,:,:);
nsubs = numel(goodsubs);

figure(2); clf
sx = ceil(nsubs / 2);
for isub = 1:nsubs
    subplot(2,sx,isub)
    
    sub = squeeze(wsub(isub,:,:,:));
    sub = (sub - mean(sub(:))) ./ std(sub(:));
    
    sd = squeeze(std(sub));
    [i,j] = find(sd == max(sd(:)));
    
%     [~, bestlag] = max(abs(sub(:,i,j)));
    extrema = max(abs(sub(:)));
    I = squeeze(sub(bestlag,:,:))';
%     I = I;
    Iw = I*wreadout(cc,isub);
    imagesc(rfs(ex).xax, rfs(ex).yax, zeros(numel(rfs(ex).xax))); hold on
    imagesc(xax, yax, Iw, [-1 1]*8*max(wreadout(cc,:)))

    set(gca, 'XTickLabel', '')
    set(gca, 'YTickLabel', '')
    title([isub max(sd(:))])
   
    xlim(rfs(ex).xax([1 end]))
    ylim(rfs(ex).yax([1 end]))
    grid on
    set(gca, 'GridColor', 'r')
    axis xy
    
    plot(rfs(ex).contours{cc}(:,1), rfs(ex).contours{cc}(:,2), 'k', 'Linewidth', 1.5)
end
colormap(gray.^1.2)

plot.fixfigure(gcf, 10, [nsubs*2 4], 'offsetAxes', false)

saveas(gcf, sprintf('Figures/2021_ucbtalk/model_%d_%d.pdf', ex, cc))

%% 2-stage LN network with 
figure(2); clf
figure(1); clf

xax = 0:.01:60; % one degree in arcmin
sf = 30;



for sigma1 = .2:.2:2
    sigma2 = 5;
    
    figure(1); clf
    %     sigma2 = 5;
    
    con = .5;
    grat = con*sin(sf*2*pi*xax/60);
    subplot(1,3,1)
    plot(xax, grat, 'k')
    hold on
    ylim([-1 1])
    xlabel('Space (arcminutes)')
    
    % 2-stage LN
    k1 = normpdf(xax, 30, sigma1);
    k2 = normpdf(xax, 30, sigma2);
    
    subplot(1,3,2)
    plot(xax, k1/max(k1), 'b'); hold on
    plot(xax, k2/max(k2), 'r')
    
    cons = 0:.1:1;
    resp1 = zeros(numel(cons),1);
    resp2 = zeros(numel(cons),1);
    
    for c = 1:numel(cons)
        grat = cons(c)*sin(sf*2*pi*xax/60);
        
        f1 = conv(grat, k1, 'same');
        n1 = max(f1,0);
        % n1 = n1 ./ max(1,sum(n1));
        
        f2 = conv(n1,k2, 'same');
        n2 = max(f2,0);
        % n2 = n2 ./ max(1,sum(n2));
        
        subplot(1,3,3)
        plot(xax, n1); hold on
        plot(xax, n2); hold off
        drawnow
        
        resp2(c) = sum(n2);
        resp1(c) = sum(n1);
    end
    
    
    figure(2);
    plot(cons, resp2, '-o'); hold on
    figure(1)
    
    plot.fixfigure(gcf, 10, [8 4], 'offsetAxes', false)
    saveas(gcf, sprintf('Figures/2021_ucbtalk/gratdemo_s1%d.pdf', sigma1))
end



