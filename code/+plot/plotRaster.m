function [x,y,m,s,bc] = plotRaster(sptimes, ev, win, bs, idx, plotit)
% Get x and y coordinates for a spikes raster and plot it (optional)
% [x,y,m,s,bc] = plotRaster(sptimes, ev, win, bs, idx, plotit)
%   makes peri-stimulus-time-histogram (PSTH) aligned to stimulus events
%   inputs
%       sptimes [n x 1] - vector of spike times
%       ev      [m x 1] - vector of event times
%       win     [1 x 2] - window around events to count spikes in
%       bs      [1 x 1] - scalar bin size
%       works in any units as long as all variable have the same units. 

% 20140415 jly wrote it
if nargin < 6
    plotit = 1;
    if nargin < 5
        idx = 1:numel(ev);
        if nargin < 4
            if nargin < 3
                win = [-mean(diff(sptimes)) 10*mean(diff(sptimes))];
                if nargin < 2
                    help eventPsth
                    return
                end
            end
            bs = diff(win)/100;
        end
    end
end

if isempty(idx)
    idx = 1:numel(ev);
end

be = win(1):bs:win(2); 
bc = be(1:end-1)+bs/2;
nbins = numel(bc);

validEvents = find(~isnan(ev));
nEvents = numel(validEvents);
tspcnt  = zeros(nEvents, nbins);

x = []; y = [];
for ii = 1:nEvents
    ievent = validEvents(ii);
    st = sptimes - ev(ievent);
    sts = st(st > win(1) & st < win(2));
    x = [x; [sts(:) sts(:)]];
    t = idx(ievent)*ones(numel(sts),1);
    y = [y; [t t+1]];
    spcnt = histc(st, be);
    
    if any(spcnt)
        tspcnt(ii,:)  = spcnt(1:nbins);
    end
end

m = mean(tspcnt)/bs;
v = var(tspcnt)/bs;
s = std(tspcnt)/nEvents/bs;

if plotit
    subplot(4,1,1:3)
    plot(x', y', 'k'); axis tight
    ylabel('trial')
    xlabel('time')
    subplot(4,1,4)
    plot(bc,m, 'k'); axis tight
    ylabel('spikerate')
    xlabel('time')
end
