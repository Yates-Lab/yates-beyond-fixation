function [lp,id] = calibration_loss(params, xy, targets, sigma)
% calculate loss for offline FaceCal calibration routine
% params:
% [scale x, scale y, rotation, ctr x, ctr y]

if nargin < 4
    sigma = 1;
end

n = size(xy,1);
th = params(3);
nx = size(targets,1);
R = [cosd(th) -sind(th); sind(th) cosd(th)];
S = [params(1) 0; 0 params(2)];
ctr = params(4:5);

xxy = [targets ones(nx,1)] * [R*S ctr(:)]';

% xy = [xy ones(n,1)]*[R*S ctr(:)]';
% xxy = targets;

Sig = [1 0; 0 1]*sigma;

logp = zeros(n,nx);

Sinv = pinv(Sig);

% calculate negative log gaussian probability 
for j = 1:nx
    xdiff = xy - xxy(j,:);
    xproj = xdiff*Sinv;
    quad = .5*sum(xproj.*xdiff,2);
    logp(:,j) = quad;
end

[lp, id] = min(logp,[],2);