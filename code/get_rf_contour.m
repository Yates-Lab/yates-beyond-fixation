function [con, ar, ctr, thresh, maxoutrf] = get_rf_contour(xi,yi,rf,varargin)
% get_rf_contour(xi,yi,rf,varargin)
% [con, ar, ctr, thresh, maxoutrf] = get_rf_contour(xi,yi,rf,varargin)
% thresh 
% upsample
% plot
ip = inputParser();
ip.addParameter('thresh', .5)
ip.addParameter('upsample', 1)
ip.addParameter('plot', false)
ip.parse(varargin{:});

rf = (rf - min(rf(:))) / (max(rf(:)) - min(rf(:)));

if numel(rf) ~= numel(xi)
    [xi, yi] = meshgrid(xi, yi);
end

if ip.Results.upsample > 1
    xi = imresize(xi, ip.Results.upsample);
    yi = imresize(yi, ip.Results.upsample);
    rf = imresize(rf, ip.Results.upsample);
end

thresh0 = ip.Results.thresh;
if thresh0 > 1
    error('get_rf_contour: threshold must be 0 < thresh < 1. use get_contour for unnormalized values')
end

thresh = .9;

[con0, ~, ~] = get_contour(xi, yi, rf, 'thresh', thresh, 'plot', ip.Results.plot);

cond = true;
while cond

    thresh = thresh - .1;
    [~, ~, ctr, maxoutrf] = get_contour(xi, yi, rf, 'thresh', thresh, 'plot', ip.Results.plot);
    cond = inpolygon(ctr(1), ctr(2), con0(:,1), con0(:,2));
    if thresh <= thresh0 
        cond = false;
    end
    if ~cond
        thresh = thresh + .1;
        [con, ar, ctr, maxoutrf] = get_contour(xi, yi, rf, 'thresh', thresh, 'plot', ip.Results.plot);
    end
        
end