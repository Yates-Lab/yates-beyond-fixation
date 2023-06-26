function [Y, binfun] = binTimesFast(st, eventTimes, binSize, keepSparse)
% BIN TIMES THE FASTEST WAY POSSIBLE IN MATLAB
% 
% Inputs:
%   st [T x 1]: spike times
%   eventTimes [N x 1]: event times
%   binSize [1 x 1]: size of bin
%   keepSparse: boolean (true: output is sparse, false: output is full [default])
% Outpus:
%   Y [nBins x 1]
%
% Example Call:
%   Y = binTimesFast(Times, eventTimes, binsize)

if nargin < 4
    keepSparse = false;
end

% conversion from time to bins
binfun = [];

[~, ~, ind1] = histcounts(st, eventTimes);
[~, ~, ind2] = histcounts(st, eventTimes+binSize);

ix = ind2+1 == ind1;
n = sum(ix);
Y = sparse(ind1(ix), ones(n,1), ones(n, 1), numel(eventTimes), 1);

if ~keepSparse
    Y = full(Y);
end