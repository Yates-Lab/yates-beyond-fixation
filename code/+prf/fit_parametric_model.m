function [beta, ci, mse, errorModeInfo] = fit_parametric_model(y, kx, ky)
% fit a parametric model of the receptive field
% Parameters
% (1) - orientation bandwidth
% (2) - orientation tuning (radians)
% (3) - spatial frequency tuning (cyc/deg)
% (4) - spatial frequency bandwidth
% (5) - gain
% (6) - offset
% [beta, ci] = fit_parametric_model(y, kx, ky)

y = y(:);
kxy = [kx(:) ky(:)];
% start with grid search over plausible parameters
n = 10;
thetas = linspace(0, 2*pi, n); % theta grid
sigmas = [.1 1 2];
As     = [.1 1 2];
rs     = [.1 .5 1 1.5 2 3 4 8];
offsets = 0;
gains   = 1; %[.1 .4 .5 1 2];

[p1, p2, p3, p4, p5, p6] = ndgrid(As, thetas, rs, sigmas, gains, offsets);

param_grid = [p1(:) p2(:) p3(:) p4(:) p5(:) p6(:)];

num = size(param_grid,1);

sserror = nan(num,1);
for i = 1:num 
    sserror(i) = nansum((y - prf.parametric_rf(param_grid(i,:),kxy)).^2);
end


[~, id] = min(sserror);

param0 = param_grid(id,:);

fun = @(params, x) prf.parametric_rf(params, x);
% paramsHat = lsqcurvefit(fun, param0, kxy, y);

% RobustWgtFun'- A weight function for robust fitting. Valid functions
%                      are 'bisquare', 'andrews', 'cauchy', 'fair', 'huber',
%                      'logistic', 'talwar', or 'welsch'. Default is '' (no
%                      robust fitting). Can also be a function handle that
%                      accepts a normalized residual as input and returns


opts.RobustWgtFun='';
[beta,resid,J,Sigma, mse, errorModeInfo] = nlinfit(kxy, y, fun, param0,opts);

% if any(isnan(J(:))) || any(isinf(J(:)))
%     error('nans in the jacobian')
% end
ci = nlparci(beta,resid,'covar',Sigma);