function [D, Bctrs] = polar_basis_cos(th, rho, m, n, endpoints)
% build a 2D cosine basis in orientation / spatial frequency space
if nargin < 5 || isempty(endpoints)
    endpoints =[.5, 2];
end

if nargin < 4 || isempty(n)
    n = 8;
end

if nargin < 3 || isempty(m)
    m = 8;
end

% orientation centers
bs = 180 / m;
mus = 0:bs:180-bs;

% sf centers
ctrs = 1:n;

% 2D basis
[xx,yy] = meshgrid(mus, ctrs);

% orientation tuning
B = von_mises_basis(th(:), xx(:));

% spatial frequency
C = raised_cosine(rho(:), yy(:), endpoints(1), endpoints(2));
D = B.*C;
    
Bctrs = [mus, invnl(ctrs,endpoints(1),endpoints(2))];

    
function out = nl(x,b, pows)
if nargin < 3
    pows = 2;
end
out = log(x./(b+1e-20))/log(pows);

function out = invnl(x,b,pows)
if nargin < 3
    pows = 2;
end
out = b*pows.^x;


function cosb = raised_cosine(x,n,b,pows)
if nargin < 2 || isempty(n)
    n = 5;
end

if nargin < 3 || isempty(b)
    b = 0.25;
end

if nargin < 4 || isempty(pows)
    pows = 2;
end

% cosine basis on a log(2) scale
%     Input:
%         x [n x 1] value at whih to
%         n [int or m x 1] number of basis functions or list of centers (after log transform)
   
if numel(n)==1
    ctrs = 0:n;
else
    ctrs = n;
end

nlin = nl(x,b,pows);
xdiff = abs(nlin(:)-ctrs(:)');

cosb = cos(max(-pi, min(pi, xdiff*pi)))/2 + .5;

    
function d = circ_diff_180(th1,th2,deg)
% Circular distance on a circle (wraps at 180)    
% 
%     INPUTS:
%         th1 - numpy array
%         th2 - numpy array
%         deg - boolean (default: True, angles in degrees)
%     
%     OUTPUT:
%         d (in units of degrees or radians, specifed by deg flag)

if nargin < 3 || isempty(deg)
    deg = true;
end

if deg
    th1 = th1/180*pi;
    th2 = th2/180*pi;
end
    
d = angle(exp( 1j * (th1-th2)*2))/2;

if deg
    d = d/pi * 180;
end

function out = von_mises_basis(x, n)
%     create von-mises basis for orientation
% 
%     INPUTS:
%         x - array like
% 
%         n - integer or array
%             if n is integer, create an evenly spaced basis of size n.
%             if n is an array, create a tent basis with centers at n
%     OUTPUTS:
%         B - array like, x, evaluated on the basis
    
if numel(n)==1
    bs = 180 / n;
    mus = 0:bs:180;
else
    mus = n;
    bs = mean(diff(unique(mus)));
end

kappa = (log(.5)/(cos(deg2rad(bs/2))-1))/2;
thetaD = x(:) - mus(:)';
out = von_mises_180(thetaD, kappa);

function y = von_mises_180(x,kappa,mu,L)
if nargin < 3 || isempty(mu)
    mu = 0;
end

if nargin < 4 || isempty(L)
    L = 0;
end

y = exp(kappa * cos(deg2rad(x-mu)).^2)/exp(kappa);
if L==1
    b0 = besseli(0, kappa);
    y = y ./ (180 * b0);
end
    
function y = deg2rad(x)
    y = x/180*pi;