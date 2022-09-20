function g = polarRF(xy, amplitude, xo, yo, sigma_x, sigma_y, offset)
% g = polarRF(xy, amplitude, th0, rho0, sigma_x, sigma_y, offset)

x=xy(:,1);
y=xy(:,2);

x = x/180*pi;
xo = xo/180.0*pi;

oriRF = (cos(x-xo).^2 - 1)/(sigma_x + 1e-10);

lpow = 10; %log base 10
sfRF = - (nl(y,lpow) - nl(yo,lpow) ).^2 / sigma_y;

g = offset + amplitude * exp(sfRF+oriRF);

function out = nl(x,b, pows)
if nargin < 3
    pows = 2;
end
out = log(x./(b+1e-20))/log(pows);