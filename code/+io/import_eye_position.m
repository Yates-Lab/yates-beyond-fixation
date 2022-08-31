function [Exp, fig] = import_eye_position(Exp, DataFolder, varargin)
% Import Eye Position for MarmoV5 session
% Arguments:
%   Exp         - Marmoview data structure
%   DataFolder  - Path where the raw eye traces live
% Optional arguments (as key/value pairs):
%   TAGSTART    - value to identify as a trial start (default: 63)
%   TAGEND      - value to identify as trial end (default: 64)
%   fid         - file id to dump to (default: 1 => dumps to command window)
% Outputs:
%   Exp         - updated data struct, now with vpx and vpx2ephys added
%   fig         - sync figure, returns empty if sync didn't work properly

ip = inputParser();
ip.addParameter('TAGSTART', 63)
ip.addParameter('TAGEND', 62)
ip.addParameter('fid', 1)
ip.addParameter('zero_mean', false)
ip.addParameter('normalize', false)
ip.parse(varargin{:});

TAGSTART = ip.Results.TAGSTART;
TAGEND = ip.Results.TAGEND;

fig = [];
fid = ip.Results.fid;
%% Loading up the VPX file as a long data stream
% Can be done later, might just use matlab eye data for now
%*****************************************************************
DDPI = 0;
EDF = 0;
VpxFiles = dir([DataFolder,filesep,'*.vpx']);
if isempty(VpxFiles)
    % try to find ddpi files if no VPX files
    VpxFiles = dir([DataFolder,filesep,'*.ddpi']);
    if isempty(VpxFiles)
        %******** if not DDPI, then try to find .edf from EyeLink
        VpxFiles = dir([DataFolder,filesep,'*edf.mat']);
        if isempty(VpxFiles)
            fprintf(fid, '\nWARNING: Error finding raw eye data file\n');
            fprintf(fid, 'USING ONLINE EYE POSITION\n');
            
            numTrials = numel(Exp.D);
            Exp.vpx = struct();
            Exp.vpx.raw = []; % time, x, y, pupil
            for iTrial = 1:numTrials
                ix = ~isnan(Exp.D{iTrial}.eyeData(:,1));
                tmp = Exp.D{iTrial}.eyeData(:,1:4);
                Exp.vpx.raw = [Exp.vpx.raw; tmp(ix,:)];
            end
            
            badsamples = (diff(Exp.vpx.raw(:,1))==0);
            Exp.vpx.raw(badsamples,:) = [];
            badsamples = Exp.vpx.raw(:,1)==0;
            Exp.vpx.raw(badsamples,:) = [];
            Exp.vpx.smo = Exp.vpx.raw;
            Exp.vpx2ephys = @(x) Exp.ptb2Ephys(x);
            
            % eliminate double samples (this shouldn't do anything)
            [~,ia] =  unique(Exp.vpx.raw(:,1));
            Exp.vpx.raw = Exp.vpx.raw(ia,:);
            
            % upsample eye traces to 1kHz
            new_timestamps = Exp.vpx.raw(1,1):1e-3:Exp.vpx.raw(end,1);
            new_EyeX = interp1(Exp.vpx.raw(:,1), Exp.vpx.raw(:,2), new_timestamps);
            new_EyeY = interp1(Exp.vpx.raw(:,1), Exp.vpx.raw(:,3), new_timestamps);
            new_Pupil = interp1(Exp.vpx.raw(:,1), Exp.vpx.raw(:,4), new_timestamps);
            bad = interp1(Exp.vpx.raw(:,1), double(Exp.vpx.raw(:,2)>31e3), new_timestamps);
            Exp.vpx.raw = [new_timestamps(:) new_EyeX(:) new_EyeY(:) new_Pupil(:)];
            
            Exp.vpx.raw(bad>0,2:end) = nan; % nan out bad sample times
            Exp.vpx.raw(:,3) = -Exp.vpx.raw(:,3) + 1;
            
            %% convert eye position to degrees
%             Exp = get_smo_eyetrace(Exp);
            
            Exp = get_smo_eyetrace(Exp);
            
            return
        else
            EDF = 1;
        end
    else
        DDPI = 1;
    end
end

%****** get order by date of files to import them
FileDates = cellfun(@datenum,{VpxFiles(:).date});
DoFileSort = [ (1:size(FileDates,2))' FileDates'];
FileSort = sortrows(DoFileSort,2); % sort them by date
%***** read and append files in order by date *********
BigN = 0;

figure(10); clf
filenum = [];
for zk = FileSort(:,1)'
    fname = VpxFiles(zk).name;
    vpx_filename = [DataFolder,filesep,VpxFiles(zk).name];
    if (DDPI == 1)
        vpx = read_ddpi.load_ddpi_file(vpx_filename);  % makes DDPI look like VPX
    elseif (EDF==1)
        vpx = read_edf.load_edf_file(vpx_filename);
    else
        vpx = read_vpx.load_vpx_file(vpx_filename);
    end
    if ~isempty(vpx)
        if ~BigN
            Exp.vpx = vpx;
        else
            if ~(DDPI || EDF)
                %******* append time to prevent overlap across files
                vpx.raw(:,1) = vpx.raw(:,1) + BigN;  % add time offset before concat
                vpx.smo(:,1) = vpx.smo(:,1) + BigN;
                vpx.tstrobes = vpx.tstrobes + BigN;
            end

            if ip.Results.zero_mean
                lost_track = vpx.raw(:,2)==max(vpx.raw(:,2)) | vpx.raw(:,3)==max(vpx.raw(:,3));
                
                vpx.raw(~lost_track,2) = vpx.raw(~lost_track,2) - mean(vpx.raw(~lost_track,2));
                vpx.raw(~lost_track,3) = vpx.raw(~lost_track,3) - mean(vpx.raw(~lost_track,3));

                if ip.Results.normalize
                    vpx.raw(~lost_track,2) = vpx.raw(~lost_track,2) / std(vpx.raw(~lost_track,2));
                    vpx.raw(~lost_track,3) = vpx.raw(~lost_track,3) / std(vpx.raw(~lost_track,3));
                end

            end
            %******* concatenate large file stream **********
            Exp.vpx.raw = [Exp.vpx.raw ; vpx.raw];
            filenum = [filenum; zk*ones(size(vpx.raw, 1), 1)];
            figure(10); 
            plot(vpx.raw(:,1), vpx.raw(:,2)); hold on
            Exp.vpx.smo = [Exp.vpx.smo ; vpx.smo];
            Exp.vpx.tstrobes = [Exp.vpx.tstrobes ; vpx.tstrobes];
            Exp.vpx.strobes = [Exp.vpx.strobes ; vpx.strobes];
            %******** compute new last time, plus one minute
        end

        BigN = Exp.vpx.smo(end,1) + 60.0; % last time point plus one minute
        
        
        clear vpx;
        fprintf(fid, '\n\n******************************************\n');
        fprintf(fid, 'Experiment file [%s] loaded\n',fname);
    else
        fprintf(fid, 'WARNING: failed to read [%s]\n',fname);
    end
end

%% Synching up the strobes from VPX to MarmoView
% Same thing, might use matlab eye data for now
%*****************************************************************
fprintf(fid, '\nSynching up vpx strobes\n');
nTrials = size(Exp.D,1);
for k = 1:nTrials
    start = synchtime.find_strobe_time(TAGSTART,Exp.D{k}.STARTCLOCK,Exp.vpx.strobes,Exp.vpx.tstrobes);
    finish = synchtime.find_strobe_time(TAGEND,Exp.D{k}.ENDCLOCK,Exp.vpx.strobes,Exp.vpx.tstrobes);
    if (isnan(start) || isnan(finish))
        fprintf(fid, 'Synching VPX trial %d/%d\n',k, nTrials);
        if isnan(finish) && isnan(start)
            fprintf(fid, 'Dropping entire VPX trial %d from protocol %s\n',k,Exp.D{k}.PR.name);
            Exp.D{k}.START_VPX = NaN;
            Exp.D{k}.END_VPX = NaN;
        else
            %******* here we could try to repair a missing code, or find a
            %******* a partial code nearby
            Exp.D{k}.START_VPX = start;
            Exp.D{k}.END_VPX = finish;
            tdiff = Exp.D{k}.eyeData(end,6) - Exp.D{k}.eyeData(1,1);
            if isnan(start) && ~isnan(finish)
                Exp.D{k}.START_VPX = Exp.D{k}.END_VPX - tdiff;
                fprintf(fid, 'Approximating VPX start code\n');
            end
            if isnan(finish) && ~isnan(start)
                Exp.D{k}.END_VPX = Exp.D{k}.START_VPX + tdiff;
                fprintf(fid, 'Approximating VPX end code\n');
            end
            %****************
        end
    else
        Exp.D{k}.START_VPX = start;
        Exp.D{k}.END_VPX = finish;
    end
end
fprintf(fid, '\nFinished synching up vpx strobes\n\n');

%% fix any issues with edf clock resetting
% for some reason, the eyelink can sometimes reset the clock at the start
% of each file. I don't know why this happens, but if it does, we need to
% sync clocks for each file separately and then combine
vpxstarts = cellfun(@(x) x.START_VPX, Exp.D);
breaks = find(diff(vpxstarts)<0);

if any(breaks)
    newt = Exp.vpx.raw(:,1); % initialize new timestamps to modify
    tbreaks = find(diff(newt) < 0);
    tstart = [1; tbreaks+1];
    tstop = [tbreaks; numel(newt)];

    startinds = [1; breaks+1];
    stopinds = [breaks; numel(vpxstarts)];

    
    nbreaks = numel(startinds);
    assert(nbreaks == numel(tstart), 'import_eye_position: different number of breaks in the file between strobe times and raw clock')
    
    for iibr = 1:nbreaks
        
        vpxstart = cellfun(@(x) x.START_VPX, Exp.D(startinds(iibr):stopinds(iibr)));
        vpxstop = cellfun(@(x) x.END_VPX, Exp.D(startinds(iibr):stopinds(iibr)));
        ptbstart = cellfun(@(x) x.STARTCLOCKTIME, Exp.D(startinds(iibr):stopinds(iibr)));
        ptbstop = cellfun(@(x) x.ENDCLOCKTIME, Exp.D(startinds(iibr):stopinds(iibr)));
        
        [vpxt, ind] = sort([vpxstart; vpxstop]);
        ptbt = [ptbstart; ptbstop];
        ptbt = ptbt(ind);

        
        [~, ifun] = synchtime.align_clocks(ptbt, vpxt);
% 
%         figure(iibr); clf
%         subplot(1,2,1)
%         plot(ptbt, vpxt, 'o'); hold on
%         plot(ptbt, fun(ptbt))
%         xlabel('PTB Clock')
%         ylabel('Eye Clock')
% 
% %         fun = synchtime.align_clocks(vpxt, ptbt);
%         subplot(1,2,2)
%         plot(vpxt, ptbt, 'o'); hold on
%         plot(vpxt, ifun(vpxt))
%         xlabel('Eye Clock')
%         ylabel('PTB Clock')

%         iix = Exp.vpx.raw(:,1) >= min(vpxt) & Exp.vpx.raw(:,1) <= max(vpxt);
        iix = tstart(iibr):tstop(iibr);
        newt(iix) = ifun(newt(iix));

%         figure()
%         plot(Exp.vpx.raw(:,1), '.'); hold on
%         plot(iix, Exp.vpx.raw(iix,1), '.')
        

    end
    
    % correct timestamps and conversion function permanently
    Exp.vpx.raw(:,1) = Exp.ptb2Ephys(newt); % pre-convert eye clock to Ephys time
    Exp.vpx.smo(:,1) = Exp.vpx.raw(:,1); % copy time to smo eye traces
    Exp.vpx2ephys = @(x) x; % then use the identity for vpx2ephys
end

if ~isfield(Exp, 'vpx2ephys')
    vpx2ephys = synchtime.sync_vpx_to_ephys_clock(Exp);
    Exp.vpx2ephys = vpx2ephys;
end

if all(isnan(Exp.vpx2ephys(Exp.vpx.raw(:,1))))
    fprintf(fid, 'WARNING: Eyetracker sync did not work\n');
    fprintf(fid, 'Going to try to manually match traces. This is slow...\n');
    
    numTrials = numel(Exp.D);
    vpx = struct();
    vpx.raw = []; % time, x, y, pupil
    for iTrial = 1:numTrials
        ix = ~isnan(Exp.D{iTrial}.eyeData(:,1));
        tmp = Exp.D{iTrial}.eyeData(:,1:4);
        vpx.raw = [vpx.raw; tmp(ix,:)];
    end
    
    badsamples = (diff(vpx.raw(:,1))==0);
    vpx.raw(badsamples,:) = [];
    vpx.raw(:,3) = 1 - vpx.raw(:,3); % flip online y
    
    nsamples = size(vpx.raw,1);
    samplematch = inf(nsamples,2);
    for isample=1:20:nsamples
        xdiff = vpx.raw(isample,2) - Exp.vpx.raw(:,2);
        ydiff = vpx.raw(isample,3) - Exp.vpx.raw(:,3);
        [minerr, id] = min(sqrt(xdiff.^2 + ydiff.^2));
        samplematch(isample,1) = id;
        samplematch(isample,2) = minerr;
    end
    
    goodsamples = find(samplematch(:,2)<1e-5);
    ptbTime = vpx.raw(goodsamples,1);
    vpxTime = Exp.vpx.raw(samplematch(goodsamples,1));
    wts = robustfit(vpxTime, Exp.ptb2Ephys(ptbTime));
    
    fig = figure(100); clf
    plot(vpxTime, Exp.ptb2Ephys(ptbTime), 'o'); hold on
    Exp.vpx2ephys = @(t) t*wts(2) + wts(1);
    plot(vpxTime, Exp.vpx2ephys(vpxTime), 'r')
    xlabel('Eye Tracker Time')
    ylabel('Ephys Time')
    legend({'Data', 'Fit'})
    set(gcf, 'PaperSize', [4 4], 'PaperPosition', [0 0 4 4])
   
    
end


%% clean up and resample
Exp.vpx.raw0 = Exp.vpx.raw;

Exp.vpx.raw(isnan(Exp.vpx.raw)) = 32000;

% detect valid epochs (this could be invalid due to failures in the
% eyetracker parameters / blinks / etc)
x = double(Exp.vpx.raw(:,2)==32000);
x(1) = 0;
x(end) = 0;

bon = find(diff(x)==1);
boff = find(diff(x)==-1);

bon = bon(2:end); % start of bad
boff = boff(1:end-1); % end of bad
if isempty(bon)
    boff = 1;
    bon = size(Exp.vpx.raw,1);
end
gdur = Exp.vpx.raw(bon,1)-Exp.vpx.raw(boff,1);

remove = gdur < 1;

bremon = bon(remove);
bremoff = boff(remove);
gdur(remove) = [];

goodSnips = sum(gdur/60);

totalDur = Exp.vpx.raw(end,1)-Exp.vpx.raw(1,1);
totalMin = totalDur/60;

fprintf(fid, '%02.2f/%02.2f minutes are usable\n', goodSnips, totalMin);

% go back through and eliminate snippets that are not analyzable
n = numel(bremon);
for i = 1:n
    Exp.vpx.raw(bremoff(i):bremon(i),2:end) = 32e3;
end

% eliminate double samples (this shouldn't do anything)
[~,ia] =  unique(Exp.vpx.raw(:,1));
Exp.vpx.raw = Exp.vpx.raw(ia,:);

% upsample eye traces to 1kHz
new_timestamps = Exp.vpx.raw(1,1):1e-3:Exp.vpx.raw(end,1);
new_EyeX = interp1(Exp.vpx.raw(:,1), Exp.vpx.raw(:,2), new_timestamps);
new_EyeY = interp1(Exp.vpx.raw(:,1), Exp.vpx.raw(:,3), new_timestamps);
new_Pupil = interp1(Exp.vpx.raw(:,1), Exp.vpx.raw(:,4), new_timestamps);
bad = interp1(Exp.vpx.raw(:,1), double(Exp.vpx.raw(:,2)>31e3), new_timestamps);
Exp.vpx.raw = [new_timestamps(:) new_EyeX(:) new_EyeY(:) new_Pupil(:)];

Exp.vpx.raw(bad>0,2:end) = nan; % nan out bad sample times


%% convert eye position to degrees
Exp = get_smo_eyetrace(Exp);

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


