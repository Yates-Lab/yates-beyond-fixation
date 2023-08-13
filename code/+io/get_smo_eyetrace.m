function Exp = get_smo_eyetrace(Exp)
Fs = 1./nanmedian(diff(Exp.vpx.raw(:,1)));
% x and y position
vxx = Exp.vpx.raw(:,2);
vyy = 1 - Exp.vpx.raw(:,3);
vtt = Exp.vpx.raw(:,1);

%--- OLD WAY: use most common calibraiton
% use the most common value across trials (we should've only calibrated
% once in these sessions)
% cx = mode(cxs);
% cy = mode(cys);
% dx = mode(dxs);
% dy = mode(dys);
% 
% % convert to d.v.a.
% vxxd = (vxx - cx)/(dx * Exp.S.pixPerDeg);
% vyy = 1 - vyy;
% vyyd = (vyy - cy)/(dy * Exp.S.pixPerDeg);

% NEW WAY
nTrials = numel(Exp.D);
validTrials = 1:nTrials;

% gain and offsets from online calibration
cxs = cellfun(@(x) x.c(1), Exp.D(validTrials));
cys = cellfun(@(x) x.c(2), Exp.D(validTrials));
dxs = cellfun(@(x) x.dx, Exp.D(validTrials));
dys = cellfun(@(x) x.dy, Exp.D(validTrials));

vxxd = vxx;
vyyd = vyy;

for iTrial = 1:nTrials
    tstartPtb = Exp.ptb2Ephys(Exp.D{iTrial}.STARTCLOCKTIME);
    tstartVpx = find(Exp.vpx2ephys(vtt) > tstartPtb, 1);
    if iTrial < nTrials
        tNextPtb = Exp.ptb2Ephys(Exp.D{iTrial+1}.STARTCLOCKTIME);
        tEndVpx = find(Exp.vpx2ephys(vtt) > tNextPtb, 1);
    else
        tEndVpx = numel(vtt);
    end
    
    iix = tstartVpx:tEndVpx;
        
    cx = cxs(1);
    cy = cys(1);
    dx = dxs(1);
    dy = dys(1);
    
    % convert to d.v.a.
    vxxd(iix) = (vxx(iix) - cx)/(dx * Exp.S.pixPerDeg);
    vyyd(iix) = (vyy(iix) - cy)/(dy * Exp.S.pixPerDeg);

end

vpp = Exp.vpx.raw(:,4);

vxx = medfilt1(vxxd, 5);
vyy = medfilt1(vyyd, 5);

vxx = imgaussfilt(vxx, 7);
vyy = imgaussfilt(vyy, 7);

vx = [0; diff(vxx)];
vy = [0; diff(vyy)];

vx = sgolayfilt(vx, 1, 3);
vy = sgolayfilt(vy, 1, 3);

% convert to d.v.a / sec
vx = vx * Fs;
vy = vy * Fs;

spd = hypot(vx, vy);
Exp.vpx.smo = [vtt vxxd vyyd vpp vx vy spd];
