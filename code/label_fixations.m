function [fstarts, fstops] = label_fixations(eyeDegX,eyeDegY,framelen,velthresh,dt,winsize)
% [fixstarts, fixstops] = label_fixations(eyeDegX,eyeDegY,framelen,velthresh,dt)

if nargin < 6 || isempty(winsize)
    winsize = 20;
end

if nargin < 5 || isempty(dt)
    dt = 1e-3;
end

if nargin < 4 || isempty(velthresh)
    velthresh = 10;
end

if nargin < 3 || isempty(framelen)
    framelen = 1;
end

% get derivative
dxdt = @(x) imgaussfilt(filter([1; -1], 1, x), 5);

if framelen > 1
    exx = sgolayfilt(eyeDegX, 1, framelen);
    eyy = sgolayfilt(eyeDegY, 1, framelen);
else
    exx = eyeDegX;
    eyy = eyeDegY;
end

% detect fixations
spd = hypot(dxdt(exx)/dt, dxdt(eyy)/dt);
fixs = spd < velthresh;
L = bwlabel(fixs);
fixs = ismember(L, find(arrayfun(@(x) x.Area > winsize, regionprops(fixs))));

fstarts = find(diff(fixs)==1);
fstops = find(diff(fixs)==-1);
if ~isempty(fstarts) && ~isempty(fstops)
    if fstarts(1) > fstops(1)
        fstarts = [1; fstarts];
    end

    if fstarts(end)>fstops(end)
        fstops = [fstops; numel(eyeDegY)];
    end
end