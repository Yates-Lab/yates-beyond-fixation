function cmap = coolwarm(n)

if nargin < 1
    n = 128;
end

% approximates the coolwarm colormap from matplotlib
c1 = [0.1216    0.4667    0.7059];
c2 = [0.8392    0.1529    0.1569];

n1 = ceil(n/2);
n2 = floor(n/2);
cmap = [ [linspace(c1(1), .9, n1); linspace(c1(2), .9, n1); linspace(c1(3), .9, n1)]'; ...
    flipud([linspace(c2(1), .9, n2); linspace(c2(2), .9, n2); linspace(c2(3), .9, n2)]')];