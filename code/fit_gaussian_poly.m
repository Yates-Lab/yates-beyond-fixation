function [params, fun] = fit_gaussian_poly(x,y,thresh)
% fit a gaussian using a polynomial trick
% If the minimum of a Gaussian function is 0, then it can be parameterized
% y = Amp*exp( - (x - mu).^2 / 2 / sigma^2)
%
% Taking the log of both sides yields the log of Amp and a quadratic
%
% ln(y) = ln(Amp)  - x.^2/2/sigma^2 + x*mu/sigma^2 - mu.^2/2/sigma^2
%
% which can be rewritten as a polynomial of form:
% ln(y) = a + bx + cx^2
%
% where
% a = ln(Amp) - mu^2/2/sigma^2
%
% b = mu/sigma^2
% 
% c = -1/2/sigma^2
%
% Thus, the gaussian parameters can be recovered quickly
%
% mu = -b/2/c
%
% sigma = sqrt(-1/2/c)
%
% Amp = exp(a - b^2/4/c) 
%
% Inputs:
%  x 
%  y
%  thresh (optional): points above this value will be fit. It's important
%                     to include this because the log of values close to
%                     zero will explode
% Outputs:
%  params
%        [mu, sigma, Amplitude, baseline]

if nargin < 3
    thresh = 0.2;
end

% normalize data between 0 and 1
mny = min(y);
mxy = max(y);
mux = mean(x);
x = x - mux;
y = (y - mny)/(mxy-mny);

use = y > thresh;

% main fitting
abc = polyfit(x(use), log(y(use)), 2);

a = abc(3);
b = abc(2);
c = abc(1);

mu = -b/(2*c);
sigma = sqrt(-1/(2*c));
amp = exp(a - b^2/(4*c));

params = [mu+mux, sigma, amp*(mxy-mny) mny];

if nargout > 1
    fun = @(params, x) params(4) + params(3)*exp( - (x - params(1)).^2/2/params(2)^2);
end

