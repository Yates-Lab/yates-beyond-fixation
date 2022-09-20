function h = raster(ii, jj, height, varargin)
% h = plotRaster(ii, jj, height, lineargs)

ii = ii(:)';
jj = jj(:)';
n = numel(jj);

if nargin < 4
    lineargs = {'k'};
else 
    lineargs = varargin;
end

if nargin < 3 
    height = 1;
end

x = [ii; ii; nan(1,n)];
y = [jj; jj+height; nan(1, n)];
    
h = plot(x(:), y(:), lineargs{:});


