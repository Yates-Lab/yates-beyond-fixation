function [stAn, stAn_SD, widx, wfs, spidx] = eventTriggeredAverage(an, st, win)
% calculate a triggered average
% [stAn, stAn_SD, widx, wfs, spidx] = eventTriggeredAverage(an, st, win)
% INPUTS
%   an [n x m] analog data (number of samples by number of channels)
%   st [k x 1] vector of event indices to trigger on
%  win [1 x 2] left and right window edges (eg. [-500 500]) 
% OUTPUTS
%       stAn - triggered average
%    stAn_SD - triggered standard deviation

%%  Spike Triggered LFP
if ~exist('win', 'var')
    win = [-500 500]; % -500 ms to +500 ms
end

widx = win(1):win(2); % -500 ms to +500 ms
windowSize = numel(widx);

stAn    = zeros(windowSize, size(an, 2));
stAn_SD = zeros(windowSize, size(an, 2));

validEvents=find(~(isnan(st) | st > size(an,1)));
% st(isnan(st)) = []; % remove impossible spikes
% st(st > size(an,1)) = [];
n=numel(st);
widxs = bsxfun(@plus, st(validEvents), widx);
bad=widxs(:,1) <= 0;
validEvents(bad)=[];
widxs(bad,:) = []; % ignore very early spikes
bad=widxs(:,end) > size(an,1);
validEvents(bad)=[];
widxs(bad,:) = []; % ignore very late spikes
spidx = all(widxs,2);
if nargout >= 4
    wfs = nan(n, numel(widx)*size(an,2));
end
    
for kCh = 1:size(an,2)
    l = an(widxs, kCh);
    l = reshape(l, [], windowSize);
    stAn(:, kCh) = nanmean(l);
    stAn_SD(:, kCh) = nanstd(l); %/ sqrt(size(l,1));
    if nargout >=4
        wfs(validEvents,(1:numel(widx)) + numel(widx) * (kCh-1)) = l;
    end
end
