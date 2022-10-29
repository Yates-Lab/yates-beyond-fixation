function out = get_spat_rf_coarsefine_reg(Exp)
% wrappper for getting the RFs at coarse and fine scale

% parameters are hard coded
out = struct();

% Coarse: get RFs across "whole screen" -- a large central region. If you
% have a bigger screen, this can go bigger.
BIGROI = [-16 -10 16 10];
binSize = 1;
stat = spat_rf_reg(Exp, 'stat', [], ...
    'debug', false, ...
    'plot', false, ...
    'ROI', BIGROI, ...
    'frate', 30, ...
    'binSize', binSize, 'spikesmooth', 0, ...
    'fitRF', false, ...
    'win', [-5 8]);

out.coarse = stat;
out.BIGROI = BIGROI;

%% Plot putative ROIs


NC = size(stat.srf,3);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));

figure(2); clf
regs = nan(NC,1);

subs = [];
for cc = 1:NC
    rf = stat.srf(:,:,cc);
    rf = (rf - min(rf(:))) / (max(rf(:)) - min(rf(:)));
    subplot(sx, sy, cc)
    imagesc(stat.xax, stat.yax, rf)
    s = regionprops(rf>.9);
    if numel(s) == 1
        subs = [subs; s];
    end
    regs(cc) = numel(s);
end



mrf = mean(stat.srf(:,:,regs==1),3);
mrf = (mrf - min(mrf(:))) / (max(mrf(:)) - min(mrf(:)));

bw = (mrf>.5);
s = regionprops(bw);

if numel(s) > 1
    bw = (mrf>.7);
    s = regionprops(bw);
end

figure(3); clf
subplot(1,3,1)
imagesc(mrf)

subplot(1,3,2)
imagesc(bw)

subplot(1,3,3)
imagesc(stat.xax, stat.yax, mrf); hold on

iix = [s.Area]>1;
if sum(iix) > 0
    s = s(iix);
end

nrois = numel(s);
binsizes = zeros(nrois,1);
rois = cell(nrois,1);
rfstats = cell(nrois,1);

for i = 1:nrois
    
    bb = s(i).BoundingBox;
    sz = max(bb(3:4))*[2 2];
    bb(1:2) = bb(1:2) - sz/4;
    bb(3:4) = sz;
%     rectangle('Position', bb , 'EdgeColor', 'r', 'Linewidth', 2)
    NEWROI = [bb(1) bb(2) bb(1) + bb(3) bb(2) + bb(4)];
    ROI = NEWROI*binSize + BIGROI([1 2 1 2]);

    bs = (ROI(3) - ROI(1)) / 20;
    plot(ROI([1 1 3 3 1]), ROI([2 4 4 2 2]), 'r', 'Linewidth', 2)

    binsizes(i) = bs;
    rois{i} = ROI;
end

%% Run analyses at a higher resolution
for i = 1:nrois

    NEWROI = rois{i};
    bs = binsizes(i);
    
    if any(isnan(NEWROI))
        disp('No Valid ROI')
    else
        rfstats{i} = spat_rf_reg(Exp, 'stat', [], ...
        'plot', false, ...
        'debug', false, ...
        'ROI', NEWROI, ...
        'binSize', bs, 'spikesmooth', 7, ...
        'r2thresh', 0.002,...
        'frate', 120, ...
        'numlags', 12, ...
        'win', [-5 15]);
    end
end

%% check which rf stats to use

out.rfstats = rfstats;
out.rois = rois;

if numel(rfstats)==1
    id = ones(1, numel(rfstats{1}.r2));
    out.fine = rfstats{1};
    out.NEWROI = rois{1}; % save one for backwards compatibility
else
    [~, id] = max(cell2mat(cellfun(@(x) x.r2', rfstats, 'uni', 0)));
    out.fine = rfstats{round(mode(id))};
    out.NEWROI = rois{round(mode(id))}; % save one for backwards compatibility
end

% loop over cells and save the best fit for each one
out.RFs = [];
for cc = 1:numel(id)

    RF = struct();
    RF.timeax = rfstats{id(cc)}.timeax;
    RF.xax = rfstats{id(cc)}.xax;
    RF.yax = rfstats{id(cc)}.yax;
    RF.roi = rfstats{id(cc)}.roi;
    RF.srf = rfstats{id(cc)}.srf(:,:,cc);
    RF.mu = rfstats{id(cc)}.rffit(cc).mu;
    RF.C = rfstats{id(cc)}.rffit(cc).C;
    RF.ar = rfstats{id(cc)}.rffit(cc).ar;
    RF.temporalPref = rfstats{id(cc)}.temporalPref(:,cc);
    RF.temporalPrefSd = rfstats{id(cc)}.temporalPrefSd(:,cc);
    RF.temporalNull = rfstats{id(cc)}.temporalNull(:,cc);
    RF.temporalNullSd = rfstats{id(cc)}.temporalNullSd(:,cc);
    RF.r2 = rfstats{id(cc)}.r2(cc);
    RF.r2rf = rfstats{id(cc)}.r2rf(cc);
    
    out.RFs = [out.RFs; RF];
end
    
