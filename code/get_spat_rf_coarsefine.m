function out = get_spat_rf_coarsefine(Exp)
% wrappper for getting the RFs at coarse and fine scale

% parameters are hard coded
out = struct();

try
    
    %%
    % spatial
    BIGROI = [-14 -8 14 8];

    stat = spat_rf_reg(Exp, 'stat', [], ...
        'debug', true, 'plot', true, ...
        'ROI', BIGROI, ...
        'frate', 30, ...
        'binSize', 1, 'spikesmooth', 0, ...
        'fitRF', false, ...
        'win', [-5 8]);
    
    out.coarse = stat;
    out.BIGROI = BIGROI;
    
    figure(3); clf
    mrf = mean(stat.srf,3);
    mrf = (mrf - min(mrf(:))) / (max(mrf(:)) - min(mrf(:)));
    
    bw = (mrf>.5);
    s = regionprops(bw);
    
    if numel(s) > 1
        bw = (mrf>.7);
        s = regionprops(bw);
    end
    
    subplot(1,3,1)
    imagesc(mrf)
    
    subplot(1,3,2)
    imagesc(bw)
    
    subplot(1,3,3)
    imagesc(mrf); hold on
%%
    [~, ind] = sort([s.Area], 'descend');
    
    for i = ind(1)
        bb = s(i).BoundingBox;
        sz = max(bb(3:4))*[2 2];
        bb(1:2) = bb(1:2) - sz/4;
        bb(3:4) = sz;
        rectangle('Position', bb , 'EdgeColor', 'r', 'Linewidth', 2)
    end
    
    roix = interp1(linspace(.5,numel(stat.xax),numel(stat.xax)), stat.xax, [bb(1) bb(1) + bb(3)]);
    roiy = interp1(linspace(.5,numel(stat.yax),numel(stat.yax)), stat.yax, [bb(2) bb(2) + bb(4)]);
    
    % if at edge
    if isnan(roix(1))
        roix(1) = stat.xax(1)-.5;
    end
    
    if isnan(roix(2))
        roix(2) = stat.xax(end)+.5;
    end
    
    if isnan(roiy(1))
        roiy(1) = stat.yax(1)-.5;
    end
    
    if isnan(roiy(2))
        roiy(2) = stat.yax(end)+.5;
    end
    
    figure(4); clf
    imagesc(stat.xax, stat.yax, mrf); hold on
    plot(roix, roiy([1 1]), 'r')
    plot(roix, roiy([2 2]), 'r')
    plot(roix([1 1]), roiy, 'r')
    plot(roix([2 2]), roiy, 'r')
    
    NEWROI = [roix(1) roiy(1) roix(2) roiy(2)];
    
    
    out.NEWROI = NEWROI;
    
    if any(isnan(NEWROI))
        disp('No Valid ROI')
    else
        stat = spat_rf_reg(Exp, 'stat', [], ...
            'plot', true, ...
            'ROI', NEWROI, ...
            'binSize', .3, 'spikesmooth', 7, ...
            'r2thresh', 0.002,...
            'frate', 120, ...
            'win', [-5 15]);
        
        out.fine = stat;
        figure(10); clf, plot(max(stat.r2rf, 0), '-o')
    end
    
catch me
    fprintf('Error finding spatial RFs for [%s]\n')
    disp(me.message)
end
