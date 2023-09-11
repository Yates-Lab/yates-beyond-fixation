function [m,s,bc,v, tspcnt] = eventPsth(sptimes, ev, win, bs, skern)
% [m, s, bc, v, tspcnt] = eventPsth(sptimes, ev, win, bs, skern)
%   makes peri-stimulus-time-histogram (PSTH) aligned to stimulus events
%   inputs
%       sptimes [n x 1] - vector of spike times
%       ev      [m x 1] - vector of event times
%       win     [1 x 2] - window around events to count spikes in
%       bs      [1 x 1] - scalar bin size
%       skern   [1 x k] - smoothing kernel
%       works in any units as long as all variable have the same units. 

% 20140415 jly wrote it

if nargin < 5
    skern = [];
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

be = win(1):bs:win(2); 
bc = be(1:end-1)+bs/2;
nbins = numel(bc);

if ~isempty(skern)
    skern = skern/sum(skern);
    znorm = filter(skern,1,ones(nbins+1,1));
end
validEvents = find(~isnan(ev));
nEvents = numel(validEvents);
tspcnt  = zeros(nEvents, nbins);
for ii = 1:nEvents
    ievent = validEvents(ii);
    st = sptimes - ev(ievent);
    spcnt = histc(st, be);
    if isempty(skern)
        smcnt = spcnt;
    else
        smcnt = filter(skern,1,spcnt)./znorm;
    end
    
    if any(spcnt)
        tspcnt(ii,:)  = smcnt(1:nbins);
    end
end

m = mean(tspcnt)/bs;
v = var(tspcnt)/bs;
s = std(tspcnt)/sqrt(nEvents)/bs;


    
