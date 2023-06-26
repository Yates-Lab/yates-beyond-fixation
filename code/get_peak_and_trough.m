function [peakloc, peak, troughloc, trough] = get_peak_and_trough(ts, wf, bnds)
% [peakloc, peak, troughloc, trough] = get_peak_and_trough(ts, wf, bnds)
if nargin < 3
    bnds = [-inf inf];
end

%--- find peak and trough
dwdt = imgaussfilt(diff(wf), 2);
    
% get trough
troughloc = findZeroCrossings(dwdt, 1);
troughloc = troughloc(ts(troughloc) > bnds(1) & ts(troughloc) < bnds(2)); % only troughs before .35 ms after clipping
tvals = wf(troughloc);
[~, mxid] = min(tvals);
troughloc = ts(troughloc(mxid));

% get peak
peakloc = findZeroCrossings(dwdt, -1);
peakloc = peakloc(ts(peakloc) < bnds(2) & ts(peakloc) > bnds(1)); % only troughs before .35 ms after clipping
pvals = wf(peakloc);
[~, mxid] = max(pvals);
peakloc = ts(peakloc(mxid));

if isempty(peakloc)
    peakloc = nan;
end

if isempty(troughloc)
    troughloc = 0;
end
    
% interpolate trough / peak with softamx
softmax = @(x,y,p) x(:)'*(y(:).^p./sum(y(:).^p));
    
winix = abs(ts - troughloc) < .02;

troughloc = softmax(ts(winix), -wf(winix), 10);
    
winix = abs(ts - peakloc) < .02;
peakloc = softmax(ts(winix), wf(winix), 10);
    
trough = interp1(ts, wf, troughloc);
peak = interp1(ts, wf, peakloc);

    
if isempty(peakloc)
    peakloc = nan;
    peak = nan;
end

if isempty(troughloc)
    troughloc = nan;
    trough = nan;
end
    