function Exp = importFreeViewingPLDAPS(S)
% import PLDAPS sessions and convert them into a marmoV5 structure

disp('FREE VIEWING IMPORT')
disp('THIS MUST BE RUN ON A COMPUTER CONNECTED TO THE MITCHELLLAB SERVER')
disp('REQUIRES MARMOPIPE CODE IN THE PATH')

% get paths
SERVER_DATA_DIR = getpref('FREEVIEWING', 'SERVER_DATA_DIR');

if contains(S.rawFilePath, SERVER_DATA_DIR)
    DataFolder = S.rawFilePath;
else
    DataFolder = fullfile(SERVER_DATA_DIR, S.rawFilePath);
end

assert(exist(DataFolder, 'dir')==7, 'importFreeViewing: raw data path does not exist')

% Load spikes data
[sp,osp] = io.import_spike_sorting(DataFolder);

if isfield(osp, 'wfs')
    osp = rmfield(osp, 'wfs');
end
% load old EPHYS session
sess = io.loadSession(DataFolder);
sess.path = DataFolder;

PDS = io.getPds(sess);

% collapse PTB2OE
ns = numel(PDS);
OEfit = zeros(2,ns);
f = functions(PDS{1}.PTB2OE);
% handle edge case
if ~isempty(strfind(f.function, 'ppval')) % this session used a spline to align clocks because there was weird slippage
    % all PDS files were combined. This *should* work
    ptb2Ephys = PDS{1}.PTB2OE;
else % proceed normally
    for i = 1:ns
        f = functions(PDS{i}.PTB2OE);
        OEfit(:,i) = f.workspace{1}.OE2PTBfit;
    end

    OE2PTBfit = mean(OEfit,2);
    ptb2Ephys = @(x) (x-OE2PTBfit(2))/OE2PTBfit(1); % function handle does nothing (we will already conver times for PLDAPS import)
end

% get eye position data
[eyeData, timestamps, ~] = io.getEyeData(sess);

%% we need to map PLDAPS stimuli to MarmoV5 stimuli

% convert PDS cell array to trial-by-trial struct array
trial = io.getPdsTrialData(PDS);

% make an Exp struct to mimic the marmoV5 output
newExp = struct();

newExp.DataFolder = sess.path;
% newExp.FileTag = [sess.subject(1) datestr(sess.dateNum, 'yyyymmdd') '.mat'];

% save spike times / clusters
newExp.osp = osp;
newExp.sp = sp;
newExp.ptb2Ephys = ptb2Ephys;

% --- convert the S struct
newExp.S.newera = PDS{1}.initialParametersMerged.newEraSyringePump.use;
if PDS{1}.initialParametersMerged.eyelink.useAsEyepos
    newExp.S.eyetracker = 'eyelink';
else
    newExp.S.eyetracker = 'mouse';
end
newExp.S.arrington = false;
newExp.S.DummyEye = false;
newExp.S.solenoid = false;
newExp.S.DummyScreen = false;
newExp.S.EyeDump = true;
newExp.S.Datapixx = true;
newExp.S.monitor = trial(1).display.displayName;
newExp.S.screenNumber = trial(1).display.scrnNum;
newExp.S.frameRate = trial(1).display.frate;
newExp.S.screenRect = trial(1).display.winRect;
newExp.S.screenWidth = trial(1).display.dWidth;
newExp.S.centerPix = trial(1).display.ctr(1:2);
newExp.S.guiLocation = [];
newExp.S.bgColour = trial(1).display.bgColor(1);
newExp.S.gamma = 1;
newExp.S.screenDistance = trial(1).display.viewdist;
newExp.S.pixPerDeg = trial(1).display.ppd;
newExp.S.TimeSensitive = [];
newExp.S.pumpCom = PDS{1}.initialParametersMerged.newEraSyringePump.port;
newExp.S.pumpDiameter = PDS{1}.initialParametersMerged.newEraSyringePump.diameter;
newExp.S.pumpRate = PDS{1}.initialParametersMerged.newEraSyringePump.rate;
try
    newExp.S.pumpDefVol = str2double(PDS{1}.initialParametersMerged.newEraSyringePump.initialVolumeGiven);
catch
    try
        newExp.S.pumpDefVol = str2double(PDS{1}.initialParametersMerged.newEraSyringePump.vol);
    catch
        newExp.S.pumpDefVol = str2double(PDS{1}.initialParametersMerged.newEraSyringePump.volume);
    end
end
newExp.S.MarmoViewVersion = 'PLDAPS';
newExp.S.finish = nan;
newExp.S.protocol = 'stimuli.forage.forage';

% --- find relevant trials from pldaps
if isfield(trial(1).pldaps, 'trialFunction')
    forageTrials = find(arrayfun(@(x) any(strfind(x.pldaps.trialFunction, 'forage')), trial, 'uni', 1));
    forageField = 'stimulus';
elseif isfield(trial, 'forage')
    forageTrials = find(arrayfun(@(x) x.forage.use, trial, 'uni', 1));
    forageField = 'forage';
end

if isfield(trial(1).pldaps, 'faceforage') % isempty(forageTrials)
   forageTrials = intersect(forageTrials, find(~arrayfun(@(x) isempty(x.faceforage), trial)));
   forageField = 'faceforage';
end

hartleyTrials = find(arrayfun(@(x) ~isempty(x.hartley), trial));
hartleyTrials = hartleyTrials(arrayfun(@(x) x.hartley.use, trial(hartleyTrials)));

forageTrials = union(hartleyTrials, forageTrials);

% if isfield(trial, 'gaussianNoiseBlobs')
%     gaussPyrTrial = find(arrayfun(@(x) ~isempty(x.gaussianNoiseBlobs), trial));
%     gaussPyrTrial = gaussPyrTrial(arrayfun(@(x) x.gaussianNoiseBlobs.use, trial(gaussPyrTrial)));
%     forageTrials = union(gaussPyrTrial, forageTrials);
% end

nTotalTrials = numel(forageTrials);

newExp.D = cell(nTotalTrials,1);

for iTrial = 1:nTotalTrials
    
    pldapsTrial = forageTrials(iTrial);
    
    % some basics
    newExp.D{iTrial}.STARTCLOCKTIME = trial(pldapsTrial).trstart;
    newExp.D{iTrial}.ENDCLOCKTIME = trial(pldapsTrial).timing.datapixxTRIALEND(1);
    newExp.D{iTrial}.STARTCLOCK = trial(pldapsTrial).unique_number;
    newExp.D{iTrial}.ENDCLOCK = nan(1,6);
    
    newExp.D{iTrial}.c = [0 0];
    newExp.D{iTrial}.dx = 1./newExp.S.pixPerDeg;
    newExp.D{iTrial}.dy = 1./newExp.S.pixPerDeg;
    
    newExp.D{iTrial}.START_EPHYS = newExp.ptb2Ephys(newExp.D{iTrial}.STARTCLOCKTIME);
    newExp.D{iTrial}.END_EPHYS = newExp.ptb2Ephys(newExp.D{iTrial}.ENDCLOCKTIME);
    
    % which protocol is it?
    if isfield(trial(pldapsTrial), 'hartley') && trial(pldapsTrial).hartley.use
        protocol = 'Grating';
    elseif isfield(trial(pldapsTrial), 'natImgBackground') && trial(pldapsTrial).natImgBackground.use
        protocol = 'BackImage';
    elseif isfield(trial(pldapsTrial), 'csdFlash') && trial(pldapsTrial).csdFlash.use
        protocol = 'CSD';
    elseif isfield(trial(pldapsTrial), 'spatialSquares') && trial(pldapsTrial).spatialSquares.use
        protocol = 'Spatial';
    elseif isfield(trial(pldapsTrial), 'gaussianNoiseBlobs') && trial(pldapsTrial).gaussianNoiseBlobs.use
        protocol = 'GaussianPyr';
    else
        protocol = 'jnk';
        newExp.D{iTrial}.PR.name = protocol;
    end
    
    
    switch protocol
        case 'BackImage'
            % BackImage
            newExp.D{iTrial}.PR.error = 0;
            newExp.D{iTrial}.PR.startTime = trial(pldapsTrial).timing.flipTimes(1,1);
            newExp.D{iTrial}.PR.imageOff = trial(pldapsTrial).timing.flipTimes(1,end);
            
            % get image file
            path0 = regexp(fileparts(trial(pldapsTrial).natImgBackground.imgDir), '/', 'split');
            dir0 = path0{end};
            
            imgIndex = trial(pldapsTrial).natImgBackground.imgIndex(trial(pldapsTrial).natImgBackground.texToDraw);
            fname0 = trial(pldapsTrial).natImgBackground.fileList(imgIndex).name;
            
            newExp.D{iTrial}.PR.imageFile = fullfile(dir0, fname0);
            newExp.D{iTrial}.PR.destRect = trial(pldapsTrial).display.winRect;
            newExp.D{iTrial}.PR.name = 'BackImage';
            
        case 'Grating' % mimic the grating trials
            newExp.D{iTrial}.PR.hNoise = nan; % no object associated (maybe we can get the pldaps one to work?)
            newExp.D{iTrial}.PR.error = 0;
            
            newExp.D{iTrial}.PR.noisetype = 1;
            newExp.D{iTrial}.PR.noiseNum = 32;
            newExp.D{iTrial}.PR.name = 'ForageProceduralNoise';
            
            % save Grating info by converting hartley stimulus to
            % orientation and spatial frequency values like what is used in
            % MarmoV5
            %
            % for future ref: you might wonder why cart2pol gets (ky,kx) as
            % input. This is because horizontal frequencies correspond to
            % vertical orientations in the image. Therefore the x and y axis
            % are flipped when using cart2pol transforms to find the
            % orientation. Another way of thinking about this is that when
            % you index an image (e.g., I[m,n]), m is indexing the rows and
            % therefore along the vertical axis, so the axes are flipped
         
            [ori, sf] = cart2pol(trial(pldapsTrial).hartley.kys, trial(pldapsTrial).hartley.kxs);
            ori = ori/pi*180; % rad2deg
            ori(ori < 0) = 180 + ori(ori < 0); % wrap 0 to 180
            
            newExp.D{iTrial}.PR.spatoris = ori;
            newExp.D{iTrial}.PR.spatfreqs = sf;
            
            frameTimes = trial(pldapsTrial).timing.flipTimes(3,:)';
            
            [oris, spatfreqs] = cart2pol(trial(pldapsTrial).hartley.ky, trial(pldapsTrial).hartley.kx);
            
            oris = oris/pi*180; % rad2deg
            oris(oris < 0) = 180 + oris(oris < 0); % wrap 0 to 180
            n = numel(oris);
            
            spatfreqs(~trial(pldapsTrial).hartley.on) = 0;
            oris(~trial(pldapsTrial).hartley.on) = ori(end);
            
            newExp.D{iTrial}.PR.NoiseHistory = [frameTimes(1:n) oris(:) spatfreqs(:)];
            
            
            % --- FROZEN TRIALS ---- %
            % --- need to add frozen sequence
            newExp.D{iTrial}.PR.frozenSequence = false;
            newExp.D{iTrial}.PR.frozenSequenceLength = nan;
                
            % version 2
            if isfield(trial(pldapsTrial).hartley, 'generativeModel') && strcmp(trial(pldapsTrial).hartley.generativeModel, 'frozen')
                
                newExp.D{iTrial}.PR.frozenSequence = true;
                newExp.D{iTrial}.PR.frozenSequenceLength = trial(pldapsTrial).hartley.sequenceFrame/4;
                
                inds = reshape(1:trial(pldapsTrial).hartley.sequenceFrame, newExp.D{iTrial}.PR.frozenSequenceLength, 4);
                tmp_ = trial(pldapsTrial).hartley.sequence.kx(inds);
                assert(all(all(bsxfun(@eq, tmp_, tmp_(:,1)))), 'something is wrong with the frozen trial')
            % version 1
            elseif isfield(trial(pldapsTrial).hartley, 'setupRNG') && strcmp(trial(pldapsTrial).hartley.setupRNG, 'frozenSequence')
                
                newExp.D{iTrial}.PR.frozenSequence = true;
                newExp.D{iTrial}.PR.frozenSequenceLength = trial(pldapsTrial).hartley.sequenceLength;
            else
                newExp.D{iTrial}.PR.frozenSequence = false;
                newExp.D{iTrial}.PR.frozenSequenceLength = nan;
            end
            
            
        case 'CSD'
            frameTimes = trial(pldapsTrial).timing.flipTimes(3,:)';
            if ~isfield(trial(pldapsTrial).csdFlash, 'on')
                % stim reconstruction requires object. We will reconstruct
                % it here
                % make on using parameters
                n = numel(frameTimes);
                on = zeros(n,1);
                frameAtChange = 1;
                isOn=false;
                
                for iFrame = 1:n
                    
                    frameCtr = iFrame - frameAtChange; % frame counter since last change
       
                    % update the flash
                    if isOn % if it's on
            
                        % is it time to turn off?
                        if frameCtr > trial(pldapsTrial).csdFlash.onDuration
                            % turn it off
                            isOn = false;
                            frameAtChange = iFrame;
                        end
                    else
            
                    % is it time to turn on?
                    if frameCtr > trial(pldapsTrial).csdFlash.offDuration    
                        % turn it off
                        isOn = true;
                        frameAtChange = iFrame;
                    end
                    
                    end
                    on(iFrame) = isOn;
                end
        
            else
                on = trial(pldapsTrial).csdFlash.on;
            end
            n = numel(on);
            
            newExp.D{iTrial}.PR.noisetype = 3;
            newExp.D{iTrial}.PR.NoiseHistory = [frameTimes(1:n) on];
            newExp.D{iTrial}.PR.name = 'ForageProceduralNoise';
        case 'GaussianPyr'
            % convert the flashed gaussian stimulus to a flashed dot
            % stimulus to work like MarmoV5 stimuli
            frameTimes = trial(pldapsTrial).timing.flipTimes(3,:)';
            newExp.D{iTrial}.P.dotSize = 0.5; % Approximate a dot size based on the gaussian size
            newExp.D{iTrial}.PR.name = 'ForageProceduralNoise';
            
            xpos = trial(pldapsTrial).gaussianNoiseBlobs.xpos - newExp.S.centerPix(1);
            ypos = -(trial(pldapsTrial).gaussianNoiseBlobs.ypos - newExp.S.centerPix(2));
            newExp.D{iTrial}.PR.noiseNum = size(xpos,2);
            
            nt = size(xpos,1);
            newExp.D{iTrial}.PR.NoiseHistory = [frameTimes(1:nt) xpos ypos];
            newExp.D{iTrial}.PR.noisetype = 5;
        case 'Spatial'
            frameTimes = trial(pldapsTrial).timing.flipTimes(3,:)';
            newExp.D{iTrial}.P.dotSize = trial(pldapsTrial).spatialSquares.size/2; % convert square to radius? hack
            newExp.D{iTrial}.PR.name = 'ForageProceduralNoise';
            
            xpos = squeeze(mean(trial(pldapsTrial).spatialSquares.pos([1 3],:,:),1))';
            ypos = squeeze(mean(trial(pldapsTrial).spatialSquares.pos([2 4],:,:),1))';
            xpos = xpos - newExp.S.centerPix(1);
            ypos = -(ypos - newExp.S.centerPix(2));
            
            newExp.D{iTrial}.PR.noiseNum = size(xpos,2);
            
            nt = size(xpos,1);
            newExp.D{iTrial}.PR.NoiseHistory = [frameTimes(1:nt) xpos ypos];
            newExp.D{iTrial}.PR.noisetype = 5;
%             xdots = NoiseHistory(iFrame,1 + (1:numDots)) + Exp.S.centerPix(1); %/Exp.S.pixPerDeg;
%             ydots = -NoiseHistory(iFrame,1 + numDots + (1:numDots)) + Exp.S.centerPix(2); %/Exp.S.pixPerDeg;
        otherwise
            
            
    end
    
    % --- track probe
    % save forage info foraging
    frameTimes = trial(pldapsTrial).timing.flipTimes(3,:)';
    n = numel(frameTimes)-1;
    if isfield(trial(pldapsTrial).(forageField), 'x')
        x = trial(pldapsTrial).(forageField).x(1:n,1);
        y = trial(pldapsTrial).(forageField).y(1:n,1);
    
        newExp.D{iTrial}.PR.ProbeHistory = [x(:) y(:) ones(n,1) frameTimes(1:n)];
    
        for i = 2:size(trial(pldapsTrial).(forageField).x,2)
            x = trial(pldapsTrial).(forageField).x(1:n,i);
            y = trial(pldapsTrial).(forageField).y(1:n,i);
            newExp.D{iTrial}.PR.ProbeHistory = [newExp.D{iTrial}.PR.ProbeHistory x y ones(n,1)*i];
        end
    else
         newExp.D{iTrial}.PR.ProbeHistory = [nan(n,1) nan(n,1) ones(n,1) frameTimes(1:n)];
    end
   
    
    % eyeData
    
    % rewardtimes
    newExp.D{iTrial}.rewardTimes = trial(pldapsTrial).behavior.reward.timeReward;
    
%     % parameters
%     newExp.D{iTrial}.P
%     
%     % protocol
%     newExp.D{iTrial}.PR
    
end

Exp = newExp;

% remove unsupported protocols
Exp.D(cellfun(@(x) strcmp(x.PR.name, 'jnk'), Exp.D)) = [];
nTotalTrials = numel(Exp.D);
%% eye position
% MEDFILT = 3;
% GSIG = 5;

% raw is flipped in marmoV5
eyeData(2,:) = -eyeData(2,:);

vpx.raw0 = [timestamps(:) eyeData'];


% % upsample eye traces to 1kHz
% new_timestamps = timestamps(1):1e-3:timestamps(end);
% new_EyeX = interp1(timestamps, eyeData(1,:), new_timestamps);
% new_EyeY = interp1(timestamps, eyeData(2,:), new_timestamps);
% new_Pupil = interp1(timestamps, eyeData(3,:), new_timestamps);
% 
% 
% vpx.raw = [new_timestamps(:) new_EyeX(:) new_EyeY(:) new_Pupil(:)];

% this gets overwritten
% vpx.smo = vpx.raw;
% vpx.smo(:,2) = medfilt1(vpx.smo(:,2),MEDFILT);
% vpx.smo(:,3) = medfilt1(vpx.smo(:,3),MEDFILT);
% vpx.smo(:,4) = medfilt1(vpx.smo(:,4),MEDFILT);
% 
% vpx.smo(:,2) = imgaussfilt(vpx.smo(:,2),GSIG);
% vpx.smo(:,3) = imgaussfilt(vpx.smo(:,3),GSIG);
% vpx.smo(:,4) = imgaussfilt(vpx.smo(:,4),GSIG);

Exp.vpx = vpx;

% clean up and resample
Exp.vpx.raw = Exp.vpx.raw0;

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

gdur = Exp.vpx.raw(bon,1)-Exp.vpx.raw(boff,1);

remove = gdur < 1;

bremon = bon(remove);
bremoff = boff(remove);
gdur(remove) = [];

goodSnips = sum(gdur/60);

totalDur = Exp.vpx.raw(end,1)-Exp.vpx.raw(1,1);
totalMin = totalDur/60;

fprintf('%02.2f/%02.2f minutes are usable\n', goodSnips, totalMin)

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


for iTrial = 1:nTotalTrials
    % hack to make the VPX time match the ephys time -- pldaps import
    % already aligns them
    Exp.D{iTrial}.START_VPX = Exp.D{iTrial}.START_EPHYS;
    Exp.D{iTrial}.END_VPX = Exp.D{iTrial}.END_EPHYS;
    ix = Exp.D{iTrial}.START_EPHYS >= Exp.vpx.raw(:,1) ...
        & Exp.D{iTrial}.END_EPHYS <= Exp.vpx.raw(:,1);
    
    Exp.D{iTrial}.eyeData = Exp.vpx.raw(ix,[1 2 3 4 1 1 1 1]);
    
end

% Saccade processing:
% Perform basic processing of eye movements and saccades
Exp = saccadeflag.run_saccade_detection_cloherty(Exp, ...
    'ShowTrials', false,...
    'accthresh', 2e4,...
    'velthresh', 10,...
    'velpeak', 10,...
    'isi', 0.02);

% track invalid samples
Exp.vpx.Labels(isnan(Exp.vpx.raw(:,2))) = 4;




disp('Done importing session');
