%% add paths
user = 'jakelaptop';
addFreeViewingPaths(user);
addpath scripts/
figDir = 'figures/fig02';
% fid = 1; % output file id (1 dumps to the command window)
fid = fopen(fullfile(figDir, 'fig02_summary.txt'), 'w');
datadir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'preprocessed');
sesslist = arrayfun(@(x) x.name, dir(fullfile(datadir, '*.mat')), 'uni', 0);

%% Loop over sesssions, get spatial receptive fields
outdir = './output/fig02_spat_reg';
 
set(groot, 'DefaultFigureVisible', 'off')

Srf = cell(numel(sesslist), 1);
overwrite = false;

for isess = 1:numel(sesslist)
    sessname = sesslist{isess};

    fname = fullfile(outdir, sessname);
    
    if exist(fname, 'file') && ~overwrite
        RFs = load(fname);

    else
        Exp = load(fullfile(datadir, sessname));

        RFs = get_spat_rf_coarsefine_reg(Exp);
        
        save(fname, '-v7.3', '-struct', 'RFs');
        
    end

    Srf{isess} = RFs;
    
end
disp("Done")
set(groot, 'DefaultFigureVisible', 'on')

%% Loop over sessions, get Grating RFs
Sgt = cell(numel(sesslist),1);
fittype = 'basis';
outdir = './output/fig02_grat_rf_reg';
overwrite = false;
for isess = 1:numel(sesslist)
    sessname = sesslist{isess};
%     fprintf('%d) %s\n\n', isess, sessname)
    fname = fullfile(outdir, sessname);
    
    if exist(fname, 'file') && ~overwrite
        RFs = load(fname);

    else
        Exp = load(fullfile(datadir, sessname));

        RFs = grat_rf_basis_reg(Exp, 'plot', false, 'debug', false);

        save(fname, '-v7.3', '-struct', 'RFs');

    end

    Sgt{isess} = RFs;
    
    
end
disp("Done")


%% Loop over sessions, measure Visual Drive / Waveform Stats
Wfs = cell(numel(sesslist),1);
outdir = './output/fig02_waveforms';
overwrite = false;
disp("")
for isess = 1:numel(sesslist)
    sessname = sesslist{isess};
    fname = fullfile(outdir, sessname);
    
    if exist(fname, 'file') && ~overwrite
        s = load(fname);
        vis = s.vis;

    else
        Exp = load(fullfile(datadir, sessname));

        evalc("vis = io.get_visual_units(Exp, 'plotit', false, 'visStimField', 'BackImage');");
        save(fname, '-v7.3', 'vis');
    end

    Wfs{isess} = vis;
    
    
end
disp("Done")

%% Make a table to summarize number of units, success rate, 

PAT = '(?<subject>\w+)_(?<date>\d{8})'; %(?<month>\d{2})(?<day>\d{2})_\.(?<ext>\w+)
spike_thresh = 200;
nboot = 500;
wfampthresh = 40;

info = cellfun(@(x) regexp(x, PAT, 'names'), sesslist, 'uni', 1);
subject = arrayfun(@(x) x.subject, info, 'uni', 0);
date = arrayfun(@(x) datestr(datenum(x.date, 'yyyymmdd'), 'mm/dd/yyyy'), info, 'uni', 0);
NumUnitsTotal = cellfun(@(x) numel(x.fine.r2), Srf);
NumUnitsSpikesThresh = cellfun(@(x,y) sum(x.fine.numspikes>spike_thresh & y.numspikes > spike_thresh), Srf, Sgt);

VisuallyDriven = cellfun(@(x) sum(arrayfun(@(y) y.BackImage.sig<.05, x)), Wfs);
HasRF = cellfun(@(x) sum(arrayfun(@(y) y.r2rf > .4, x.RFs)), Srf);
HasLinRF = cellfun(@(x) sum(arrayfun(@(y) y.r2 > 0, x.RFs)), Srf);

DurationGratings = cellfun(@(x) x.numsamples/x.frate, Sgt);
DurationSpatialDots = cellfun(@(x) x.fine.nsamples/x.fine.frate, Srf);
T = table(subject, date, NumUnitsTotal, NumUnitsSpikesThresh, VisuallyDriven, HasRF, HasLinRF, HasGratRF, DurationGratings, DurationSpatialDots);


m = median(DurationGratings);
mci = bootci(nboot, @median, DurationGratings);
fprintf(fid, "Grating stimulus was run for %2.2f [%2.2f, %2.2f] seconds\n", m, mci(1), mci(2));
fprintf(fid, "Minimum duration was %2.2f seconds\n\n", min(DurationGratings));

m = median(DurationSpatialDots);
mci = bootci(nboot, @median, DurationSpatialDots);
fprintf(fid, "Spatial mapping stimulus was run for %2.2f [%2.2f, %2.2f] seconds\n", m, mci(1), mci(2));
fprintf(fid, "Minimum duration was %2.2f seconds\n\n", min(DurationSpatialDots));

writetable(T, fullfile(figDir, 'summarytable.xls'))
txt = evalc('disp(T)');
fprintf(fid, txt);

%% Example units

exs = {'logan_20191231.mat', 'ellie_20190111.mat'};
ccs = [19, 9];


for ii = 1:numel(exs)
    ex = find(strcmp(sesslist, exs{ii}));
    cc = ccs(ii);
    
    figure(1); clf
    
    t = tiledlayout(3,1);
    t.TileSpacing = 'compact';
    
    ax1 = nexttile;
    I = Srf{ex}.coarse.srf(:,:,cc);
    I = (I - min(I(:))) ./ (max(I(:)) - min(I(:)));
    imagesc(Srf{ex}.coarse.xax, Srf{ex}.coarse.yax, I); hold on
    axis xy
    
    NEWROI = Srf{ex}.RFs(cc).roi;
    plot(NEWROI([1 1 3 3 1]), NEWROI([2 4 4 2 2]))
    roi = [NEWROI([1 2]) NEWROI([3 4])-NEWROI([1 2])] + .15;
    
    rectangle('Position', roi, 'EdgeColor', 'r', 'Linewidth', 1)
    
    I = Srf{ex}.fine.srf(:,:,cc);
    I = (I - min(I(:))) ./ (max(I(:)) - min(I(:)));
    
    pos = ax1.Position;
    pos(1:2) = pos(1:2) + [.1 0.05];
    
    aspect = 1/2;
    pos(3) = pos(3)/4;
    pos(4) = pos(3)*aspect;
    ax2 = axes('Position', pos);
    
    
    imagesc(Srf{ex}.fine.xax, Srf{ex}.fine.yax, I); hold on
    plot([0 0], Srf{ex}.fine.yax([1 end]), 'Color', 'k')
    plot(Srf{ex}.fine.xax([1 end]), [0 0], 'Color', 'k')
    
    cmap = plot.coolwarm;
    colormap(cmap)
    
    if isfield(Srf{ex}.fine, 'contours')
        plot(Srf{ex}.fine.contours{cc}(:,1), Srf{ex}.fine.contours{cc}(:,2), 'r')
    end

    if isfield(Srf{ex}.fine.rffit(cc), 'C')
        plot.plotellipse(Srf{ex}.fine.rffit(cc).mu, Srf{ex}.fine.rffit(cc).C, 1, 'k', 'Linewidth', 1);
    end
    axis xy
    
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
    axis tight

    xlabel('Time lag (ms)')
    ylabel('Firing Rate (sp s^{-1})')
    
    plot.offsetAxes(ax)
    set(gca, 'XTick', -40:40:120)
    
    plot.fixfigure(gcf, 8, [2.5 6], 'offsetAxes', false)
    ax2.XColor = [1 0 0];
    ax2.YColor = [1 0 0];
    ax2.Box = 'on';
    saveas(gcf, fullfile(figDir, sprintf('example_%s_%d.pdf', strrep(sesslist{ex}, '.mat', ''), cc)))
    
end

%% Loop over SRF struct and get relevant statistics
r2 = []; % r-squared from gaussian fit to RF
cvr2 = []; % cross-validated r-squared for linear rf
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


numspikesS = [];
numspikesG = [];

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
        
    numspikesS = [numspikesS; Srf{ex}.coarse.numspikes(:)];
    numspikesG = [numspikesG; Sgt{ex}.numspikes(:)];

    NC = numel(Srf{ex}.fine.rffit);
    for cc = 1:NC
        if ~isfield(Srf{ex}.fine.rffit(cc), 'mu')
            fprintf('Skipping because no fit %s\n', sesslist{ex})
            continue
        end
        
        
         mu = Srf{ex}.RFs(cc).mu;
         C = Srf{ex}.RFs(cc).C;
         
         if isempty(Sgt{ex}.rffit(cc).r2) || isempty(Srf{ex}.fine.rffit(cc).r2)
             fprintf('Skipping because bad fit [%s] %d\n', sesslist{ex}, cc)
             wf = [wf; Wfs{ex}(cc).waveform];
             oriPref = [oriPref; nan];
             oriBw = [oriBw; nan];
             sfPref = [sfPref; nan];
             sfBw = [sfBw; nan];
             gtr2 = [gtr2; nan];
             
             r2 = [r2; nan]; % store r-squared
             cvr2 = [cvr2; nan];
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
        
         r2 = [r2; Srf{ex}.RFs(cc).r2rf]; % store r-squared
         cvr2 = [cvr2; Srf{ex}.RFs(cc).r2]; % store r-squared
         ar = [ar; Srf{ex}.RFs(cc).ar];
%          ar = [ar; Srf{ex}.fine.conarea(cc)];
%          ecc = [ecc; hypot(Srf{ex}.fine.conctr(cc,1), Srf{ex}.fine.conctr(cc,2))];
         ecc = [ecc; hypot(mu(1), mu(2))];
         mshift = [mshift; Srf{ex}.fine.rffit(cc).mushift];
         
         ctr = [ctr; [numel(r2) numel(gtr2)]];
         
         cgs = [cgs; Srf{ex}.fine.cgs(cc)];
         
         wf = [wf; Wfs{ex}(cc).waveform];
         sess = [sess; sesslist{ex}];
         
         exnum = [exnum; ex];
         
         if ctr(end,1) ~= ctr(end,2)
             keyboard
         end
    end
end


oriPref(oriPref < 0) = 180 + oriPref(oriPref < 0);
oriPref(oriPref > 180) = oriPref(oriPref > 180) - 180;

wfamp = arrayfun(@(x) x.peakval - x.troughval, wf);
isiV = arrayfun(@(x) x.isiV, wf);

validwf = wfamp > wfampthresh;
spikeixS = validwf & numspikesS > spike_thresh; % only include units (single and multi) with an amplitude > 40 microvolts
spikeixG = validwf & numspikesG > spike_thresh; 

fprintf(fid, '%d (Spatial) and %d (Grating) of %d/%d recorded units (Total) had > %d spikes and waveform amplitudes > %d microvolts\n', sum(spikeixS), sum(spikeixG), sum(validwf), numel(sigs), spike_thresh, wfampthresh);



%% Rosa comparison Eccentricity plot

% get index
validwf = wfamp > wfampthresh & numspikesS > spike_thresh; % only include units (single and multi) with an amplitude > 40 microvolts

six = r2 > 0.4 & validwf;
% six = sigs & validwf;
fprintf(fid, '%d/%d (%2.2f%%) units selective for space\n', sum(six), sum(validwf), 100*sum(six)/sum(validwf));

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


fprintf(fid, 'Rosa Fit:\n');
fprintf(fid, 'A = %02.2f\n', b0(1));
fprintf(fid, 'B = %02.2f\n', b0(2));
fprintf(fid, 'C = %02.2f\n', b0(3));

fprintf(fid, 'OUR Fit:\n');
fprintf(fid, 'A = %02.2f [%02.2f, %02.2f]\n', bhat(1), bhatci(1,1), bhatci(1,2));
fprintf(fid, 'B = %02.2f [%02.2f, %02.2f]\n', bhat(2), bhatci(2,1), bhatci(2,2));
fprintf(fid, 'C = %02.2f [%02.2f, %02.2f]\n', bhat(3), bhatci(3,1), bhatci(3,2));

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
fprintf(fid, 'r-squared for fit %02.2f and rosa %02.2f\n', r2fit, r2rosa);

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


%% Orientation Preference
addpath ~/Dropbox/MatlabCode/Repos/circstat-matlab/
oriPref(angle(oriBw)~=0) = nan; % ignore untuned units

validwf= wfamp > 40 & numspikesG > spike_thresh;
gix = sigg==1 & ~isnan(oriPref) & angle(oriBw)==0 & oriBw < 90;
gix = gix & validwf;
fprintf(fid, '%d/%d units selective for Gratings\n', sum(gix), sum(validwf)); 


figure(1); clf
t = tiledlayout(1,2);
t.TileSpacing = 'compact';
nexttile


h = histogram(wrapTo180(oriPref(gix)), 'binEdges', 0:20:180, 'FaceColor', .1*[1 1 1]); hold on
text(105, .7*max(h.Values), sprintf('n=%d', sum(gix)))
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

fprintf(fid, 'Orientation distribution is significantly different than uniform:\n');
fprintf(fid, 'Circ_otest pval: %d (%02.4f)\n', pval, pval);

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
r2fit = rsquared(obw, fun(phat, op));

% plot.fixfigure(gcf, 7, [2 2], 'FontName', 'Arial', ...
%     'LineWidth',.5, 'OffsetAxes', false);

plot.formatFig(gcf, [4 1.5], 'nature')
saveas(gcf, fullfile(figDir, 'fig02_Orientation.pdf'))


%% test for difference between cardinal and oblique
bs = 45; % window around 0, 45, 90, 135, 180
bedges = -bs/2:bs:180+bs/2;
[cnt, ~, id] = histcounts(op, bedges);
obix = mod(id,2)==0;
cardix = mod(id,2)~=0;
figure(1); clf
histogram(obw(obix), 'binEdges', 0:10:180); hold on
histogram(obw(cardix), 'binEdges', 0:10:180); hold on

fprintf(fid, 'Oblique median bandwidth: %02.3f [%02.3f, %02.3f]\n', median(obw(obix)), bootci(nboot, @median, obw(obix)));
fprintf(fid, 'Cardinal median bandwidth: %02.3f [%02.3f, %02.3f]\n', median(obw(cardix)), bootci(nboot, @median, obw(cardix)));

[pval, ~, stats] = ranksum(obw(obix), obw(cardix));
fprintf(fid, 'Two-sided rank sum test: p=%d (%02.10f), ranksum=%d, zval=%d\n', pval, pval, stats.ranksum, stats.zval);


%% Separate by monkey
monk = cellfun(@(x) x(1), sesslist);
[~,monkeyId] = find(monk==unique(monk)');

cmap = lines;
six = r2 > .5;
figure(1); clf
for ex = 1:numel(sesslist)
    sessname = strrep(sesslist{ex}, '.mat', '');
    disp(sessname)
    ix = six & ex==exnum;
    plot(ecc(ix), ar(ix), '.', 'Color', cmap(monkeyId(ex),:)); hold on
end

%% plot spatial RF locations

mus(mus(:,1) < -5,1) = - mus(mus(:,1) < -5,1);

figure(10); clf
for ii = find(six)'
    plot.plotellipse(mus(ii,:), reshape(Cs(ii,:), [2 2]), 1); hold on
end

xlim([-14 14])
ylim([-10 10])

%% 
figure(10); clf
ix = six & gix;
plot(ecc(ix), sfPref(ix), 'o')
xlabel('Eccentricity (d.v.a.)')
ylabel('SF Prefrerence (cycles/deg)')


%% plot session by session
figure(10); clf
cmap = lines(max(exnum));
for ex = unique(exnum(:))'
    ii = ex==exnum(:) & gix;
    plot(sfPref(ii), sfBw(ii), '.', 'Color', cmap(ex,:)); hold on; %'wo', 'MarkerFaceColor', .2*[1 1 1], 'MarkerSize', 2)
end
xlim([0 10])
ylim([0 10])

fclose(fid)

return


%% plot session-by-session check

for isess = 1

    sessname = strrep(sesslist{isess},'.mat', '');
    disp(sessname)
    
    if isempty(Srf{isess})
        continue
    end

    NC = numel(Srf{isess}.coarse.numspikes);
    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));


    figure(1); clf
    ax = plot.tight_subplot(sx, sy, 0, 0, 0);

    for cc = 1:NC
        set(gcf, 'currentaxes', ax(cc))
        imagesc(Srf{isess}.coarse.xax, Srf{isess}.coarse.yax, Srf{isess}.coarse.srf(:,:,cc)); hold on
        plot(Srf{isess}.RFs(cc).roi([1 1 3 3 1]), Srf{isess}.RFs(cc).roi([2 4 4 2 2]), 'r', 'Linewidth', 2)
        axis off
        plot([0 0], ylim, 'y')
        plot(xlim, [0 0], 'y')
    end

    set(gcf, 'PaperSize', [sx*2 sy*1.5], 'PaperPosition', [0 0 sx*2 sy*1.5])
    saveas(gcf, fullfile('figures/fig02/sesscheck', sprintf('%s_rfcoarse.pdf', sessname)))


    figure(2); clf
    ax = plot.tight_subplot(sx, sy, 0, 0, 0);

    for cc = 1:NC
        set(gcf, 'currentaxes', ax(cc))
        imagesc(Srf{isess}.RFs(cc).xax, Srf{isess}.RFs(cc).yax, Srf{isess}.RFs(cc).srf);
        hold on
        if ~isempty(Srf{isess}.RFs(cc).contour)
            plot(Srf{isess}.RFs(cc).contour(:,1), Srf{isess}.RFs(cc).contour(:,2), 'r')
        end
%         if ~isempty(Srf{isess}.RFs(cc).mu)
%             plot.plotellipse(Srf{isess}.RFs(cc).mu, Srf{isess}.RFs(cc).C, 1, 'r');
%         end
        axis off
    end

    set(gcf, 'PaperSize', [sx*2 sy*1.5], 'PaperPosition', [0 0 sx*2 sy*1.5])
    saveas(gcf, fullfile('figures/fig02/sesscheck', sprintf('%s_rffine.pdf', sessname)))

    figure(3); clf
    ax = plot.tight_subplot(sx, sy, 0.01, 0.01, 0.01);

    for cc = 1:NC
        set(gcf, 'currentaxes', ax(cc))
        plot.errorbarFill(Srf{isess}.RFs(cc).timeax, Srf{isess}.RFs(cc).temporalPref, Srf{isess}.RFs(cc).temporalPrefSd, 'b'); hold on
        plot.errorbarFill(Srf{isess}.RFs(cc).timeax, Srf{isess}.RFs(cc).temporalNull, Srf{isess}.RFs(cc).temporalNullSd, 'r'); hold on
        plot([0 0], ylim, 'k')
        plot(xlim, [0 0])
        axis off
    end

    set(gcf, 'PaperSize', [sx*2 sy*1.5], 'PaperPosition', [0 0 sx*2 sy*1.5])
    saveas(gcf, fullfile('figures/fig02/sesscheck', sprintf('%s_rftemporal.pdf', sessname)))
end

%% Summary table
set(groot, 'DefaultFigureVisible', 'on')
