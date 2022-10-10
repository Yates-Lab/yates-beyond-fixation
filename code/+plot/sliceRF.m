function varargout = sliceRF(rf, varargin)

ip = inputParser();
ip.addParameter('dt', 20)
ip.parse(varargin{:});

nlags = size(rf,1);
NX = size(rf, 3);
NY = size(rf, 2);

extrema = max(abs(rf(:)));

xd = 1:NX;
yd = 1:NY;

dt = ip.Results.dt;

[xx,yy] = meshgrid(xd, yd);
xoffsets = linspace(0, 200, nlags);
yoffsets = linspace(20, 0, nlags)-20;
h = [];

corners = zeros(nlags, 3);
for ilag = 1:nlags
    I = squeeze(rf(ilag, :,:));
    xax = xx+xoffsets(ilag);
    yax = zeros(size(xx))+ilag*dt;
    zax = yy+yoffsets(ilag);
    h = surf(xax, yax, zax, I); hold on
    corners(ilag,:) = [min(xax(:)) min(yax(:)) min(zax(:))];
end

if nargout > 0
    varargout{1} = corners;
end

shading flat
colormap(plot.coolwarm)
axis equal
view(50, 15)
% 
% colormap gray
shading flat
caxis([-1 1]*extrema)
axis off