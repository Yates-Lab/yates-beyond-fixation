function [H, newalpha] = benjaminiYekutieli(pvals, alpha)
% [H, newalpha] = benjaminiYekutieli(pvals, alpha)
% performs a False Discovery Rate analysis on pvalues for a specified alpha
% as described here: https://en.wikipedia.org/wiki/False_discovery_rate

% (c) 2021 Jacob Yates

if nargin < 2
    alpha = 0.05;
end

% rank order pvalues
[newp, ind] = sort(pvals);

n = numel(ind);

% find deviations from a line specified by alpha
c = sum(1./(1:n));
k = find(newp > alpha*(1:n)/n/c, 1, 'first');

% reject the null hypothesis for p(1:(k-1)), or, p < p(k)
if isempty(k)
    newalpha = min(pvals); % set so that all pvalues are rejected
else
    newalpha = newp(k);
end

H = pvals < newalpha;