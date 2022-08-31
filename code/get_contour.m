function [xy, ar, ctr] = get_contour(xi,yi,rf,varargin)
% get_contour(xi,yi,rf,varargin)

ip = inputParser();
ip.addParameter('thresh', 4)
ip.addParameter('plot', false)
ip.parse(varargin{:});

rfthresh = rf > ip.Results.thresh; %100*median(rfthresh(:));

if ~any(rfthresh(:))
    xy = nan(1,2);
    ar = nan;
    ctr = nan(1,2);
    return
end
bw = bwlabel(rfthresh);
bws = unique(bw); bws(bws==0)=[];
n = numel(bws);
v = zeros(n,1);
for i = 1:n
    v(i) = sum(bws(i)==bw(:));
end
[~, ind] = max(v);
ind = bws(ind);

c = contour(xi, yi, bw==ind, 1);
cmask = contour(bw==ind, 1);

cinds = find(cmask(1,:)<=1);
if numel(cinds)<2
    cinds = [cinds size(c,2)+1];
else
    cinds = [cinds size(c,2)+1];
    [~, id] = max(diff(cinds));
    cinds = cinds(id+[0 1]);
end

cinds = cinds(1)+1:cinds(2)-1;
x = c(1,cinds);
y = c(2,cinds);
xy = [x(:) y(:)];

mask = poly2mask(cmask(1,cinds), cmask(2,cinds), size(rf,1), size(rf,2));
wts = rf.*mask;
wts = wts / sum(wts(:));
xc = xi(:)'*wts(:);
yc = yi(:)'*wts(:);

if ip.Results.plot
    figure(1); clf
    imagesc(xi(:), yi(:), rf); hold on
    plot(x, y, 'k', 'Linewidth', 2)
    plot(xc, yc, 'or')
end

ar = polyarea(x,y);
ctr = [xc yc];