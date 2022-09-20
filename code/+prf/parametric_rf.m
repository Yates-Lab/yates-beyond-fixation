function lambda = parametric_rf(params, X, logGauss, vm01)
% parametric receptive field for grating stimulus
% rate = prf.parametric_rf(params, X, logGauss)
% Inputs:
%   params [1 x 6] parameters of the model
%   X    [n x 2]   orientation,spatial frequency of each grating
%
% Outpus:
%   rate [n x 1] rate at each kx,ky point
%
%  Parameters are:
%   1. Orientation Kappa
%   2. Orientation Preference
%   3. Spatial Frequency Preference
%   4. Spatial Frequency Sigma
%   5. Gain
%   6. Offset

if nargin  < 4
    vm01 = true; % backwards compatibility
end

if nargin < 3 || isempty(logGauss)
    logGauss = true;
end

orientation = X(:,1);
orientation(isnan(orientation)) = 0;
spatialFrequency = X(:,2);


% Von Mises that wraps at pi and normalized between 0 and 1
if params(1)==0
    orientationTuning = ones(size(orientation));
else
    if vm01
        orientationTuning = (exp(params(1)*cos(orientation - params(2)).^2) - 1) / (exp(params(1)) - 1);
    else
        orientationTuning = (exp(params(1)*cos(orientation - params(2)).^2) - params(1));
    end
end

if logGauss == 1
    % LOG GAUSSIAN
    spatialFrequencyTuning = exp( - ( (log(1 + spatialFrequency) - log(1 + params(3))).^2/2/params(4)^2));
    % spatialFrequencyTuning = exp( - ( (log(spatialFrequency) - log(params(3))).^2/2/params(4)^2));
else
    % RAISED COSINE
    logbase = log(max(params(4), 1.1));
    spatialFrequencyTuning = cos( min(max( (log(spatialFrequency)/logbase)-log(params(3))/logbase, -pi/2), pi/2));
end

lambda = params(6) + (params(5) - params(6)) * (orientationTuning .* spatialFrequencyTuning);
% lambda = params(5)*exp(orientationTuning - spatialFrequencyTuning) + params(6);


% y = minval + (maxval - minval) * (exp(k*(cos(x - pref)+1)) - 1) / (exp(2*k) - 1);