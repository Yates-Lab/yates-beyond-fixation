function out = get_spat_rf_coarsefine(Exp)
% wrappper for getting the RFs at coarse and fine scale

% parameters are hard coded
out = struct();

% Coarse: get RFs across "whole screen" -- a large central region. If you
% have a bigger screen, this can go bigger.
BIGROI = [-16 -10 16 10];
binSize = .5;
stat = spat_rf(Exp, 'ROI', BIGROI, ...
    'binSize', binSize);

out.coarse = stat;
out.BIGROI = BIGROI;

%%
cc = 1;

%%
cc = cc + 1;
NC = numel(stat.maxV);
if cc > NC
    cc = 1;
end

% cc = 1;
figure(1); clf
imagesc(stat.xax, stat.yax, stat.srf(:,:,cc)); hold on
title(stat.rfsig(cc))
% if ~isempty
% plot(stat.contours{})

%%
NC = numel(stat.maxV);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(1); clf
for cc = 1:NC
    subplot(sx, sy, cc)
    imagesc(stat.srf(:,:,cc))
end


%% Plot putative ROIs

% include putative significant RFs
six = find(stat.maxV(:) > 10);

NC = numel(six);

binSize = stat.xax(2)-stat.xax(1);
x0 = min(stat.xax);
y0 = min(stat.yax);


% subplot(1,2,1)
% plot(stat.contours{cc}(:,1), stat.contours{cc}(:,2), 'r')
% xlim(stat.xax([1 end]))
% ylim(stat.yax([1 end]))

mask = zeros(stat.dim);
for i = 1:NC
    cc = six(i);
    
    cx = (stat.contours{cc}(:,1)-x0)/binSize;
    cy = (stat.contours{cc}(:,2)-y0)/binSize;
    try
        mask = mask + poly2mask(cx, cy, size(stat.srf,1), size(stat.srf,2));
    end

end

mask = mask / NC;

figure(2); clf
imagesc(stat.xax, stat.yax, mask)
hold on

bw = bwlabel(mask > .1);
s = regionprops(mask > .1);
n = max(bw(:));
mx = zeros(n,1);
mn = zeros(n,1);
for g = 1:n
    mx(g) = max(reshape(mask.*(bw==g), [], 1));
    mn(g) = sum(bw(:)==g(:));
end

iix = [s.Area]'>2 & mx>2;
if sum(iix) > 0
    s = s(iix);
end

nrois = numel(s);
binsizes = zeros(nrois,1);
rois = cell(nrois,1);


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
rfstats = cell(nrois,1);
for i = 1:nrois

    NEWROI = rois{i};
    bs = binsizes(i);
    
    if any(isnan(NEWROI))
        disp('No Valid ROI')
    else
        rfstats{i} = spat_rf(Exp, ...
        'ROI', NEWROI, ...
        'binSize', bs, ...
        'rfthresh', .5);
    end
end

%%
out.fine = rfstats;
out.rois = rois;

if numel(rfstats)==1
    NC = numel(rfstats{1}.rfsig);
    id = ones(1, NC);
    cids = 1:NC;
else
    
    NC = numel(rfstats{1}.rfsig);
    cids = 1:NC;
%     rfsig = cell2mat(cellfun(@(x) x.rfsig(:)', rfstats(:), 'uni', 0));
    maxV = cell2mat(cellfun(@(x) x.maxV(:)', rfstats(:), 'uni', 0));
    maxV(maxV > 500) = 0;
    
    [~, id] = max(maxV);
end

% loop over cells and save the best fit for each one
out.RFs = [];
for cc = 1:NC
    RF = struct();
    RF.cid = cids(cc);
    RF.timeax = rfstats{id(cc)}.timeax;
    RF.xax = rfstats{id(cc)}.xax;
    RF.yax = rfstats{id(cc)}.yax;
    RF.roi = rfstats{id(cc)}.roi;
    RF.srf = rfstats{id(cc)}.srf(:,:,cc);
    RF.mu = rfstats{id(cc)}.conctr(cc,:);
    RF.ar = rfstats{id(cc)}.conarea(cc);
    RF.temporalPref = rfstats{id(cc)}.temporalPref(:,cc);
    RF.temporalNull = rfstats{id(cc)}.temporalNull(:,cc);
    RF.rfsig = rfstats{id(cc)}.rfsig(cc);
    RF.maxV = rfstats{id(cc)}.maxV(cc);
    RF.peaklagt = rfstats{id(cc)}.peaklagt(cc);
    RF.contour = rfstats{id(cc)}.contours{cc};
    RF.nsamples = rfstats{id(cc)}.nsamples;
    RF.frate = rfstats{id(cc)}.frate;
    RF.numspikes = rfstats{id(cc)}.numspikes(cc);

    out.RFs = [out.RFs; RF];
end
    
