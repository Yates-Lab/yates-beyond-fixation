%% add paths
user = 'jakelaptop';
addFreeViewingPaths(user);
addpath scripts/
figDir = 'figures/fig02';

datadir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'preprocessed');
sesslist = arrayfun(@(x) x.name, dir(fullfile(datadir, '*.mat')), 'uni', 0);

%%

Exp = load(fullfile(datadir, sesslist{5}));

%%
vis = io.get_visual_units(Exp, 'plotit', true, 'visStimField', 'BackImage');

%%
field = 'Dots';
plot(arrayfun(@(x) x.(field).isiFr, vis), arrayfun(@(x) x.(field).stimFr, vis), 'o');
sig = arrayfun(@(x) x.(field).sig, vis)>0;
mfr = arrayfun(@(x) x.(field).stimFr, vis);
isiV = arrayfun(@(x) x.isiRate, vis);

figure(1); clf
plot(mfr, isiV, 'o')
hold on
plot(mfr(sig), isiV(sig), 'o')

%%


figure(1); clf;
plot(arrayfun(@(x) x.(field).isiFr, vis), arrayfun(@(x) x.(field).stimFr, vis), 'o');
sig = arrayfun(@(x) x.(field).sig, vis)>0;
mfr = arrayfun(@(x) x.(field).stimFr, vis);
isiV = arrayfun(@(x) x.isiRate, vis);

hold on
plot(arrayfun(@(x) x.(field).isiFr, vis(sig)), arrayfun(@(x) x.(field).stimFr, vis(sig)), 'o');
plot(xlim, xlim, 'k')

%%
% tmp = get_spat_rf_coarsefine(Exp);
% tmp = get_spatial_rfs(Exp);

dotTrials = io.getValidTrials(Exp, 'Dots');
if ~isempty(dotTrials)
    
    BIGROI = [-1 -.5 1 .5]*14;
    eyePos = Exp.vpx.smo(:,2:3);

    binSize = .25;
    Frate = 60;
    [Xstim, RobsSpace, opts] = io.preprocess_spatialmapping_data(Exp, ...
        'ROI', BIGROI*Exp.S.pixPerDeg, 'binSize', binSize*Exp.S.pixPerDeg, ...
        'eyePosExclusion', 500, ...
        'eyePos', eyePos, 'frate', Frate, ...
        'fastBinning', false);
end

%%
win = [-1 10];
inds = hypot(opts.eyePosAtFrame(:,1), opts.eyePosAtFrame(:,2)) < 500;
stasFull = forwardCorrelation(Xstim, RobsSpace, win, inds, [], true, true);
% stasNull = forwardCorrelation(Xstim, Shuffle(RobsSpace), win, [], [], [], true);

%%

Ssta = [];
NC = numel(vis);
thresh = .7;
for cc = 1:NC

    stas = stasFull(:,:,cc); % ./ imgaussfilt(std(stasNull(:,:,cc)), 1);
    [~, peaklag] = max(std(stas, [], 2));
    rf = reshape(stas(peaklag, :), opts.dims);
    Stmp.Im = rf;
    Stmp.contour = [nan nan];
    Stmp.contourConv = [nan nan];
    Stmp.thresh = nan;
    Stmp.ctr = [nan nan];
    Stmp.area = nan;
    Stmp.areaConvex = nan;
    Stmp.areaRatio = nan;
    Stmp.maxoutrf = nan;

    if peaklag == 1
        Ssta = [Ssta; Stmp];
        continue
    end
        
       
    [xx,yy] = meshgrid(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg);
    rf = (rf - min(rf(:))) / (max(rf(:)) - min(rf(:)));
    [con, ar, ctr, maxoutrf] = get_contour(xx,yy,rf, 'thresh', thresh);
%     [con, ar, ctr, thresh, maxoutrf] = get_rf_contour(xx,yy,rf, 'thresh', .7, 'plot', false);
    
    if thresh == .9
        Ssta = [Ssta; Stmp];
        % no contour found
        continue
    end
    

    k = convhull(con(:,1), con(:,2));
    arConv = polyarea(con(k,1), con(k,2));

    Stmp.contour = con;
    Stmp.contourConv = [con(k,1), con(k,2)];
    Stmp.thresh = thresh;
    Stmp.ctr = ctr;
    Stmp.area = ar;
    Stmp.areaConvex = arConv;
    Stmp.areaRatio = ar / arConv;
    Stmp.maxoutrf = maxoutrf;


    Ssta = [Ssta; Stmp];
   
end

%%

cc = cc+1;
if cc > NC
    cc = 1;
end
figure(1); clf
imagesc(xx(1,:), yy(:,1), Ssta(cc).Im)
hold on
plot(Ssta(cc).contour(:,1), Ssta(cc).contour(:,2), 'r')
title(Ssta(cc).maxoutrf)


figure(10); clf
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
for cc = 1:NC
    subplot(sx, sy, cc, 'align')
    imagesc(xx(1,:), yy(:,1), Ssta(cc).Im)
    hold on
    plot(Ssta(cc).contour(:,1), Ssta(cc).contour(:,2), 'r')
    title(Ssta(cc).maxoutrf/Ssta(cc).thresh)
    axis off
end

%%
sig = arrayfun(@(x) x.(field).sig, vis)>0;
mfr = arrayfun(@(x) x.(field).stimFr, vis);
isiV = arrayfun(@(x) x.isiRate, vis);


cx = arrayfun(@(x) x.ctr(1), Ssta);
cy = arrayfun(@(x) x.ctr(2), Ssta);
ar = arrayfun(@(x) x.area, Ssta);
arC = arrayfun(@(x) x.areaConvex, Ssta);
ecc = hypot(cx, cy);
thresh = arrayfun(@(x) x.thresh, Ssta);
maxoutrf = arrayfun(@(x) x.maxoutrf, Ssta);
hasrf = arrayfun(@(x) x.areaRatio, Ssta) > .9;
hasrf = maxoutrf./thresh < 1.05;

ix = hasrf;


figure(1); clf
subplot(1,2,1)
% plot(ecc, arC, 'o');
hold on
plot(ecc, ar, '.');
% plot(ecc(ix), arC(ix), 'o')
plot(ecc(ix), ar(ix), '.')
set(gca, 'xscale', 'log')
subplot(1,2,2)
plot(ar, arC, 'o');
hold on
plot(xlim, xlim, 'k')
plot(ar(ix), arC(ix), 'o');



%%


cc = cc + 1;
% stas = forwardCorrelation(Xstim, RobsSpace(:,cc), win);
% close all
figure(1); clf
stas = stasFull(:,:,cc); % ./ imgaussfilt(std(stasNull(:,:,cc)), 1);
[~, peaklag] = max(std(stas, [], 2));

% stas = stasFull(:,:,cc); % - ;
% stas = stas / std(stas(:)) - mean(stas(:));
wm = [min(stas(:)) max(stas(:))];
nlags = size(stas,1);
% try
    for ilag = 1:nlags
        subplot(2, ceil(nlags/2), ilag)
        imagesc(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg, reshape(stas(ilag, :), opts.dims), wm)
        if ilag==peaklag
            title(sprintf('peak lag: %02.2f', (win(1)+ilag-1)*16))
        else
            title(sprintf('lag: %02.2f', (win(1)+ilag-1)*16))
        end
        axis xy
    end
% end
drawnow

%%
ilag = 4;
figure(2); clf
I = reshape(stas(ilag, :), opts.dims);
% I = reshape(std(stas), opts.dims);
[xx,yy] = meshgrid(opts.xax/Exp.S.pixPerDeg, opts.yax/Exp.S.pixPerDeg);
% [con, ar, ctr, thresh] = get_rf_contour(xx,yy, I, 'plot', true)

figure(1); clf
I = (I - min(I(:))) / (max(I(:)) - min(I(:)));
subplot(1,2,1)
[con90, ar90, ctr90] = get_contour(xx, yy, I, 'thresh', .9, 'plot', false);
imagesc(xx(1,:),yy(:,1),I);
hold on
plot(con90(:,1), con90(:,2), 'r')

subplot(1,2,2)
[con50, ar50, ctr50] = get_contour(xx, yy, I, 'thresh', .5, 'plot', false);
imagesc(xx(1,:),yy(:,1),I);
hold on
plot(con50(:,1), con50(:,2), 'r')
k = convhull(con50(:,1), con50(:,2));
arC = polyarea(con50(k,1), con50(k,2));
disp(ar50/arC)
plot(con50(k,1), con50(k,2), 'g')

cond = inpolygon(ctr90(1), ctr90(2), con50(:,1), con50(:,2));
disp(cond)
%% Loop over examples, get Srf

clear Srf

sesslist = io.dataFactory;
sesslist = sesslist(1:57);

sfname = fullfile('Data', 'spatialrfsreg.mat');
Srf = cell(numel(sesslist),1);

if exist(sfname, 'file')==2
    disp('Loading Spatial RFs')
    load(sfname)
else
    rng(1234)
    for iEx = 1:numel(sesslist)
        if isempty(Srf{iEx})

            Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
            
            tmp = get_spat_rf_coarsefine(Exp);
            tmp.sorter = sorter;
            Srf{iEx} = tmp;
        end
    end
    
    save(sfname, '-v7.3', 'Srf')
end
disp("Done")

%% Grating RFs
Sgt = cell(numel(sesslist),1);
fittype = 'basis';
gfname = fullfile('Data', sprintf('gratrf_%s.mat', fittype));
if exist(gfname, 'file')==2
    disp('Loading Grating RFs')
    load(gfname)
else

    for iEx = 1:numel(sesslist)
        if isempty(Sgt{iEx})
            try
                Exp = io.dataFactoryGratingSubspace(sesslist{iEx}, 'spike_sorting', sorter);
                
                tmp = grat_rf_basis(Exp, 'plot', true, 'debug', false);
                tmp.sorter = sorter;
                drawnow
                
                Sgt{iEx} = tmp;
                
            catch me
                disp('ERROR ERROR')
                disp(me.message)
            end
        end
    end
    
    save(gfname, '-v7.3', 'Sgt')
end
disp("Done")

%% get waveform stats
Waveforms = cell(numel(sesslist), 1);
wname = fullfile('Data', 'waveforms.mat');
if exist(wname, 'file')==2
    disp('Loading Waveforms')
    load(wname)
else
    
    for iEx = 1:numel(sesslist)
        if ~isempty(Srf{iEx})
            Exp = io.dataFactory(sesslist{iEx}, 'spike_sorting', Srf{iEx}.sorter);
            Waveforms{iEx} = io.get_waveform_stats(Exp.osp);
        end
    end
    save(wname, '-v7.3', 'Waveforms')
    
end

disp("Done")
%% Example units
exs = [25, 27, 45, 45, 5]; % session #
ccs = [9, 8, 19, 35, 14]; % unit #

% ccs = 6;
% exs = 10*ones(1,numel(ccs));
% ex = ex - 1;
% cc = 0;
% sesslist{ex}
% %%
% cc = cc + 1;
% if cc > size(Srf{ex}.coarse.rf,3)
%     cc = 1;
% end
for ii = 1:numel(exs)
    ex = exs(ii);
    cc = ccs(ii);
    
    figure(1); clf
    
    t = tiledlayout(3,1);
    t.TileSpacing = 'compact';
    
    ax1 = nexttile;
    I = Srf{ex}.coarse.srf(:,:,cc);
    I = (I - min(I(:))) ./ (max(I(:)) - min(I(:)));
    imagesc(Srf{ex}.coarse.xax, Srf{ex}.coarse.yax, I); hold on
    axis xy
    
    roi = [Srf{ex}.NEWROI([1 2]) Srf{ex}.NEWROI([3 4])-Srf{ex}.NEWROI([1 2])] + .15;
    
    rectangle('Position', roi, 'EdgeColor', 'r', 'Linewidth', 1)
    % offset = [0 0];
    % offset = [-5 -2];
    % roi(1:2) = roi(1:2) + offset;
    % rectangle('Position', roi, 'EdgeColor', 'r', 'Linewidth', 2)
    I = Srf{ex}.fine.srf(:,:,cc);
    I = (I - min(I(:))) ./ (max(I(:)) - min(I(:)));
    % I = flipud(I);
    % imagesc(Srf{ex}.fine.xax+offset(1), fliplr(Srf{ex}.fine.yax)+offset(2), I);
    % plot(mean(Srf{ex}.fine.xax) + offset(1)*[1 1], Srf{ex}.fine.yax([1 end])+offset(2), 'Color', 'k')
    % plot(Srf{ex}.fine.xax([1 end])+offset(1), mean(Srf{ex}.fine.yax) + offset(2)*[1 1], 'Color', 'k')
    % axis xy
    
    
    pos = ax1.Position;
    pos(1:2) = pos(1:2) + [.1 0.05];
    
    aspect = 1/2; %roi(3)./roi(4);
%     aspect = 1;
    pos(3) = pos(3)/4;
    pos(4) = pos(3)*aspect;
    ax2 = axes('Position', pos);
    
    
    imagesc(Srf{ex}.fine.xax, Srf{ex}.fine.yax, I); hold on
    plot([0 0], Srf{ex}.fine.yax([1 end]), 'Color', 'k')
    plot(Srf{ex}.fine.xax([1 end]), [0 0], 'Color', 'k')
    
    % plot(offset(1)*[1 1], Srf{ex}.fine.yax([1 end])+offset(2), 'Color', 'k')
    
    cmap = plot.coolwarm;
    colormap(cmap)
    
    % if Srf{ex}.fine.sig(cc)
    if isfield(Srf{ex}.fine.rffit(cc), 'mu')
        plot.plotellipse(Srf{ex}.fine.rffit(cc).mu, Srf{ex}.fine.rffit(cc).C, 1, 'k', 'Linewidth', 1);
    end
    axis xy
    % hold off
    
    ax2.XTickLabel = [];
    ax2.YTickLabel = [];
    
    set(gcf, 'currentaxes', ax1)
    
    plot(xlim, [0 0], 'k')
    plot([0 0], ylim, 'k')
    xlabel('Azimuth (d.v.a.)')
    ylabel('Elevation (d.v.a.)')
    title(cc)
    
    
    nexttile
    plot.polar_contourf(Sgt{ex}.xax, Sgt{ex}.yax, Sgt{ex}.srf(:,:,cc), 'maxrho', 8); hold on
    level = .5*max(Sgt{ex}.rffit(cc).srfHat(:));
    plot.polar_contour(Sgt{ex}.xax, Sgt{ex}.yax, Sgt{ex}.rffit(cc).srfHat, 'vmin', level, 'vmax', level, 'nlevels', 1, 'grid', 'off', 'maxrho', 8)
    
    ax = nexttile;
    plot.errorbarFill(Sgt{ex}.timeax(:), Sgt{ex}.temporalPref(:,cc), Sgt{ex}.temporalPrefSd(:,cc)); %
    hold on
    plot(Sgt{ex}.timeax(:), Sgt{ex}.temporalPref(:,cc), 'k')
    axis tight %, 'k', 'FaceColor', 'k', 'EdgeColor', 'k', 'FaceAlpha', .5); hold on
%     plot.errorbarFill(Sgt{ex}.timeax(:), Sgt{ex}.temporalNull(:,cc), Sgt{ex}.temporalNullSd(:,cc), 'k', 'FaceColor', cmap(1,:), 'EdgeColor', cmap(1,:), 'FaceAlpha', .5); hold on
    xlabel('Time lag (ms)')
    ylabel('Firing Rate (sp s^{-1})')
    
    plot.offsetAxes(ax)
    set(gca, 'XTick', -40:40:120)
    
%     set(gcf, 'PaperSize', [2.5 6], 'PaperPosition', [0 0 2.5 6])
    plot.fixfigure(gcf, 8, [2.5 6], 'offsetAxes', false)
    ax2.XColor = [1 0 0];
    ax2.YColor = [1 0 0];
    ax2.Box = 'on';
    saveas(gcf, fullfile(figDir, sprintf('example_%s_%d.pdf', sesslist{ex}, cc)))
    
end

%% Loop over SRF struct and get relevant statistics
r2 = []; % r-squared from gaussian fit to RF
ar = []; % sqrt area (computed from gaussian fit)
ecc = []; % eccentricity

sfPref = []; % spatial frequency preference
sfBw = [];
oriPref = [];
oriBw  = [];
sigg = [];
sigs = [];

gtr2 = []; % r-squared of parametric fit to frequecy RF

ctr = [];
mus = [];
Cs = [];
cgs = [];
mshift = [];
sess = {};

wf = [];

exnum = [];

zthresh = 8;
for ex = 1:numel(Srf)
    
    if ~isfield(Srf{ex}, 'fine') || ~isfield(Sgt{ex}, 'rffit') || (numel(Sgt{ex}.rffit) ~= numel(Srf{ex}.fine.rffit))
        continue
    end
    
    if sum(Srf{ex}.fine.sig)<2 && sum(Sgt{ex}.sig)<2
        continue
    end
        
    NC = numel(Srf{ex}.fine.rffit);
    for cc = 1:NC
        if ~isfield(Srf{ex}.fine.rffit(cc), 'mu')
            fprintf('Skipping because no fit %s\n', sesslist{ex})
            continue
        end
        
        
         mu = Srf{ex}.fine.rffit(cc).mu;
         C = Srf{ex}.fine.rffit(cc).C;
         
         if isempty(Sgt{ex}.rffit(cc).r2) || isempty(Srf{ex}.fine.rffit(cc).r2)
             fprintf('Skipping because bad fit [%s] %d\n', sesslist{ex}, cc)
             wf = [wf; Waveforms{ex}(cc)];
             oriPref = [oriPref; nan];
             oriBw = [oriBw; nan];
             sfPref = [sfPref; nan];
             sfBw = [sfBw; nan];
             gtr2 = [gtr2; nan];
             
             r2 = [r2; nan]; % store r-squared
             ar = [ar; nan];
             ecc = [ecc; nan];
             sigs = [sigs; false];
             sigg = [sigg; false];
             mus = [mus; nan(1,2)];
             Cs = [Cs; nan(1,4)];
             cgs = [cgs; nan];
             mshift = [mshift; nan];
             sess = [sess; sesslist{ex}];
             exnum = [exnum; ex];
             continue
         end
         
         mus = [mus; mu];
         Cs = [Cs; C(:)'];
         
         oriPref = [oriPref; Sgt{ex}.rffit(cc).oriPref];
         oriBw = [oriBw; Sgt{ex}.rffit(cc).oriBandwidth];
         sfPref = [sfPref; Sgt{ex}.rffit(cc).sfPref];
         sfBw = [sfBw; Sgt{ex}.rffit(cc).sfBandwidth];
         gtr2 = [gtr2; Sgt{ex}.rffit(cc).r2];
             
         sigs = [sigs; Srf{ex}.fine.sig(cc)];
         sigg = [sigg; Sgt{ex}.sig(cc)];
        
         r2 = [r2; Srf{ex}.fine.r2rf(cc)]; % store r-squared
         ar = [ar; Srf{ex}.fine.rffit(cc).ar];
         ecc = [ecc; Srf{ex}.fine.rffit(cc).ecc];
         mshift = [mshift; Srf{ex}.fine.rffit(cc).mushift];
         
         ctr = [ctr; [numel(r2) numel(gtr2)]];
         
         cgs = [cgs; Srf{ex}.fine.cgs(cc)];
         
         wf = [wf; Waveforms{ex}(cc)];
         sess = [sess; sesslist{ex}];
         
         exnum = [exnum; ex];
         
         if ctr(end,1) ~= ctr(end,2)
             keyboard
         end
    end
end

% wrap
% wrap orientation
oriPref(oriPref < 0) = 180 + oriPref(oriPref < 0);
oriPref(oriPref > 180) = oriPref(oriPref > 180) - 180;

fprintf('%d (Spatial) and %d (Grating) of %d Units Total are significant\n', sum(sigs), sum(sigg), numel(sigs))

ecrl = arrayfun(@(x) x.ExtremityCiRatio(1), wf);
ecru = arrayfun(@(x) x.ExtremityCiRatio(2), wf);
wfamp = arrayfun(@(x) x.peakval - x.troughval, wf);

%% Rosa comparison Eccentricity plot
% get index
validwf = wfamp > 40; % only include units with an amplitude > 40 microvolts

six = sigs==1 & mshift < 1.25;

six = six & validwf;
fprintf('%d/%d units selective for space\n', sum(six), sum(validwf))

eccx = .1:.1:20; % eccentricity for plotting fits

% Receptive field size defined as sqrt(RF Area)
% rosa_fit = exp( -0.764 + 0.495 * log(eccx) + 0.050 * log(eccx) .^2); % rosa 1997 marmoset
% rosaHat = exp( -0.764 + 0.495 * log(x) + 0.050 * log(x) .^2); % rosa 1997 marmoset
 
figure(2); clf
x = ecc(six);
scaleFactor = sqrt(-log(.5))*2; % scaling to convert from gaussian SD to FWHM
y = ar(six) * scaleFactor;

hPlot = plot(x, y, 'o', 'Color', .8*[1 1 1], 'MarkerFaceColor', .2*[1 1 1], 'MarkerSize', 1.5, 'Linewidth', .25); hold on



b0 = [0.764, 0.495 ,0.050]; % initialize with Rosa fit
fun = @(p,x) exp( -p(1) + p(2)*log(x) + abs(p(3))*log(x).^2);

options = optimset(@lsqcurvefit);
options.Display ='none';
% [bhat,RESNORM,resid,~,~,~,J] = robustlsqcurvefit(fun, b0, x, y, [], [], [], options);
[bhat,RESNORM,resid,~,~,~,J] = lsqcurvefit(fun, b0, x, y, [], [], options);
bhatci = nlparci(bhat, resid, 'jacobian', J);


fprintf('Rosa Fit:\n')
fprintf('A = %02.2f\n', b0(1))
fprintf('B = %02.2f\n', b0(2))
fprintf('C = %02.2f\n', b0(3))

fprintf('OUR Fit:\n')
fprintf('A = %02.2f [%02.2f, %02.2f]\n', bhat(1), bhatci(1,1), bhatci(1,2))
fprintf('B = %02.2f [%02.2f, %02.2f]\n', bhat(2), bhatci(2,1), bhatci(2,2))
fprintf('C = %02.2f [%02.2f, %02.2f]\n', bhat(3), bhatci(3,1), bhatci(3,2))

hold on
cmap = lines;
[ypred, delta] = nlpredci(fun, eccx, bhat, resid, 'Jacobian', J);
plot.errorbarFill(eccx, ypred, delta, 'k', 'FaceColor', cmap(1,:), 'EdgeColor', 'none', 'FaceAlpha', .25);
hPlot(2) = plot(eccx, fun(bhat,eccx), 'Color', cmap(1,:));
hPlot(3) = plot(eccx, fun(b0,eccx), 'Color', cmap(5,:));

xlim([0 20])
ylim([0 20])

r2rosa = rsquared(y,fun(b0,x));
r2fit = rsquared(y, fun(bhat, x));
fprintf('r-squared for fit %02.2f and rosa %02.2f\n', r2fit, r2rosa)

set(gca, 'xscale', 'log', 'yscale', 'log')
xlabel('Eccentricity (d.v.a)')
ylabel('RF size (d.v.a)')
set(gcf, 'Color', 'w')
hLeg = legend(hPlot, {'Data', 'Polynomial Fit', 'Rosa 1997'}, 'Box', 'off');
hLeg.ItemTokenSize = hLeg.ItemTokenSize/6;
pos = hLeg.Position;
hLeg.Position(1) = pos(1) - .1;
hLeg.Position(2) = pos(2) - .05;
hLeg.FontName = 'Helvetica';
hLeg.FontSize = 5;

set(gca, 'XTick', [.1 1 10], 'YTick', [.1 1 10])
xt = get(gca, 'XTick');
set(gca, 'XTickLabel', xt)
yt = get(gca, 'YTick');
set(gca, 'YTickLabel', yt)

text(.25, 5, sprintf('n=%d', sum(six)))

% plot.fixfigure(gcf, 7, [2 2], 'FontName', 'Arial', ...
%     'LineWidth',.5, 'OffsetAxes', false);
plot.formatFig(gcf, [1.75 1.5], 'nature')



saveas(gcf, fullfile(figDir, 'fig02_ecc_vs_RFsize.pdf'))


%% Orientation
addpath ~/Dropbox/MatlabCode/Repos/circstat-matlab/
oriPref(angle(oriBw)~=0) = nan; % ignore untuned units
gix = sigg==1 & ~isnan(oriPref) & angle(oriBw)==0 & oriBw < 90;
gix = gix & validwf;
fprintf('%d/%d units selective for Gratings\n', sum(gix), sum(validwf))


figure(1); clf
t = tiledlayout(1,2);
t.TileSpacing = 'compact';
nexttile


h = histogram(wrapTo180(oriPref(gix)), 'binEdges', 0:10:180, 'FaceColor', .1*[1 1 1]); hold on
text(105, .7*max(h.Values), sprintf('n=%d', sum(gix)))
% xlabel('Orientation Preference ')
ylabel('Count')
set(gca,'XTick', 0:45:180, 'YTick', 0:25:50)
plot.offsetAxes(gca, true, 0)
xlabel('Orientation Preference (deg)')
ylabel('Count')

ax = nexttile;
plot(wrapTo180(oriPref(gix)), oriBw(gix), 'wo', 'MarkerFaceColor', [1 1 1]*.1, 'MarkerSize', 2); hold on
obw = oriBw(gix);  % orientation bandwidth
op = oriPref(gix); % orientation preference
be = h.BinEdges;
be2 = be(2:end);
be = be(1:end-1);
obws = arrayfun(@(x,y) mean(obw(op>x & op<y)), be,be2);
obwe = arrayfun(@(x,y) std(obw(op>x & op<y))/sqrt(sum(op>x & op<y)), be,be2);
pval = circ_otest(op/180*pi);

fprintf('Orientation distribution is significantly different than uniform:\n')
fprintf('Circ_otest pval: %d (%02.4f)\n', pval, pval)

cmap = [0 0 0];
bc = (be + be2)/2;
plot.errorbarFill(bc, obws, obwe, 'k', 'FaceColor', cmap(1,:), 'FaceAlpha', .25, 'EdgeColor', 'none');
plot((be + be2)/2, obws, 'Color', cmap(1,:))
xlim([0 180])
ylim([0 90])

xlabel('Orientation Preference (deg)')
ylabel('Orientation Bandwidth (deg)')

set(ax, 'XTick', 0:45:180, 'YTick', 0:30:90)

fun = @(params,x) params(1)*cosd(params(4)*(x+params(3))).^2 + params(2);
% fun = @(params,x) params(1)*cosd(params(4)*x+params(3)).^2 + params(2);
par0 = [10 mean(obw) 0 0];
options = optimset(@lsqcurvefit);
options.Display = 'none';
    
phat = lsqcurvefit(fun, par0, op, real(obw), [0 0 0 0], [max(obw), max(obw), 180, 2], options);
% plot(0:180, fun(phat, 0:180), 'b')
r2 = rsquared(obw, fun(phat, op));

% plot.fixfigure(gcf, 7, [2 2], 'FontName', 'Arial', ...
%     'LineWidth',.5, 'OffsetAxes', false);

plot.formatFig(gcf, [4 1.5], 'nature')
saveas(gcf, fullfile(figDir, 'fig02_Orientation.pdf'))


%% test for difference between cardinal and oblique
fid = 1;
bs = 45;
bedges = -bs/2:bs:180+bs/2;
[cnt, ~, id] = histcounts(op, bedges);
obix = mod(id,2)==0;
cardix = mod(id,2)~=0;
figure(1); clf
histogram(obw(obix), 'binEdges', 0:10:180); hold on
histogram(obw(cardix), 'binEdges', 0:10:180); hold on

fprintf(fid, 'Oblique median bandwidth: %02.3f [%02.3f, %02.3f]\n', median(obw(obix)), bootci(500, @median, obw(obix)))
fprintf(fid, 'Cardinal median bandwidth: %02.3f [%02.3f, %02.3f]\n', median(obw(cardix)), bootci(500, @median, obw(cardix)))

[pval, ~, stats] = ranksum(obw(obix), obw(cardix));
fprintf(fid, 'Two-sided rank sum test: p=%d (%02.10f), ranksum=%d, zval=%d\n', pval, pval, stats.ranksum, stats.zval)

%% plot spatial RF locations

mus(mus(:,1) < -5,1) = - mus(mus(:,1) < -5,1);

figure(10); clf
for ii = find(six)'
    plot.plotellipse(mus(ii,:), reshape(Cs(ii,:), [2 2]), 1); hold on
end

xlim([-14 14])
ylim([-10 10])

%% plot session by session
figure(10); clf
cmap = lines(max(exnum));
for ex = unique(exnum(:))'
    ii = ex==exnum(:) & gix;
    plot(sfPref(ii), sfBw(ii), '.', 'Color', cmap(ex,:)); hold on; %'wo', 'MarkerFaceColor', .2*[1 1 1], 'MarkerSize', 2)
end
xlim([0 10])
ylim([0 10])

%% Spatial Frequency by session
clf
plot(exnum(gix), sfPref(gix), '.')
ylim([0 10])


%% 
