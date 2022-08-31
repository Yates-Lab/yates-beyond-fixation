function Exp = importFreeViewing(S)
% this function imports the data required for the FREEVIEWING project
% It shares the same fundamentals as 
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
[sp,osp] = io.import_spike_sorting(DataFolder, S.spikeSorting);
io.copy_spikes_from_server(strrep(S.processedFileName, '.mat', ''), S.spikeSorting)

% Baic marmoView import. Synchronize with Ephys if it exists
Exp = io.basic_marmoview_import(DataFolder);

% fix spatial frequency bug: marmoV5 data before 2021 February had a
% scaling bug where all spatial frequencies off by a factor of two
datebugfixed = datenum('20210201', 'yyyymmdd');
thisdate = datenum(regexp(S.processedFileName, '[0-9]*', 'match', 'once'), 'yyyymmdd');
if thisdate < datebugfixed
    warning('importFreeViewing: fixing early MarmoV5 spatial frequency bug')
    Exp = io.correct_spatfreq_by_half(Exp);
end

Exp.sp = sp;  % Jude's cell array of spike times
Exp.osp = osp; % keep the original Kilosort spike format

% Import eye position signals
Exp = io.import_eye_position(Exp, DataFolder);

% clean up and resample
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


% Saccade processing:
% Perform basic processing of eye movements and saccades
Exp = saccadeflag.run_saccade_detection_cloherty(Exp, ...
    'ShowTrials', false,...
    'accthresh', 2e4,...
    'velthresh', 10,...
    'velpeak', 10,...
    'isi', 0.02);

% track invalid sampls
Exp.vpx.Labels(isnan(Exp.vpx.raw(:,2))) = 4;

validTrials = io.getValidTrials(Exp, 'Grating');
for iTrial = validTrials(:)'
    if ~isfield(Exp.D{iTrial}.PR, 'frozenSequence')
        Exp.D{iTrial}.PR.frozenSequence = false;
    end
end
        
disp('Done importing session');

