function polar_contour(th, rho, I, varargin)
% polar_contour(xx, yy, I, varargin)

ip = inputParser();
ip.addParameter('grid', 'on')
ip.addParameter('vmin', [])
ip.addParameter('vmax', [])
ip.addParameter('nlevels', 20)
ip.addParameter('Color', 'k')
ip.addParameter('TextOffset', 1)
ip.addParameter('maxrho', [])
ip.parse(varargin{:});

if isempty(ip.Results.vmin)
    vmin = min(I(:));
else
    vmin = ip.Results.vmin;
end

if isempty(ip.Results.vmax)
    vmax = max(I(:));
else
    vmax = ip.Results.vmax;
end

I(1) = vmin;
I(2) = vmax;

levels = linspace(vmin,vmax, ip.Results.nlevels);
if  ip.Results.nlevels==1
     levels = [0 levels];
end

if isempty(ip.Results.maxrho)
    maxrho = max(rho(:));
else
    maxrho = ip.Results.maxrho;
end

rhogrid = [0 2.^(0:log2(maxrho))];

mask = rho < max(rhogrid);
x = rho(mask).*cosd(th(mask));
y = rho(mask).*sind(th(mask));
I = I(mask);

[szy,szx] = find(mask, 1, 'last');
sz = [szy szx];
x = reshape(x, sz);
y = reshape(y, sz);
I = reshape(I, sz);

contour(x,y,I, levels, 'Color', ip.Results.Color, 'Linewidth', 1);


hold on
if strcmp(ip.Results.grid, 'on')
    plot.polar_grid(0:45:max(th(:)), rhogrid, 'TextOffset', ip.Results.TextOffset)
end