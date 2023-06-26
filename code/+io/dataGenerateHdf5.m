function h5name = dataGenerateHdf5(Exp, S, varargin)
% simple spike-triggered analysis of neuron responses
% simpleSTAanalyses(Exp, S, varargin)
% 
% Inputs:
%   Exp [struct]: marmoview experiment struct
%   S   [struct]: options struct from dataFactory
%       Latency: latency of the monitor
%       rect:  analysis ROI (in pixels, centered on gaze)
% Outputs:
%   several figures that are saved to disk
%
% Optional Arguments:
%   'stimulus': {'Gabor', 'Grating', 'Dots', 'BackImage', 'All'}
%   'testmode': logical. save out test set.
%   'eyesmooth': smoothing window for eye position (sgolay fitler)
%   'includeProbe': whether to include the probe in the reconstruction

% varargin = {'t_downsample', 2, 's_downsample', 2, 'eyesmooth', 7};

ip = inputParser();
ip.addParameter('stimulus', 'Gabor')
ip.addParameter('testmode', true)
ip.addParameter('eyesmooth', 41)
ip.addParameter('eyesmoothorder', 3)
ip.addParameter('includeProbe', true)
ip.addParameter('correctEyePos', [])
ip.addParameter('GazeContingent', true)
ip.addParameter('nonlinearEyeCorrection', false)
ip.addParameter('overwrite', false)
ip.addParameter('usePTBdraw', true)
ip.addParameter('debug', false)
ip.addParameter('validTrials', [])
ip.parse(varargin{:})

%% manual adjustment to rect
if ~isfield(S, 'rect')
    error('dataGenerateHdf5: you must specify a stimulus rect for analysis')
end

% get stimulus ROI
rect = S.rect;

% get clusters to analyze
if isfield(S, 'cids')
    cluster_ids = S.cids;
else
    cluster_ids = Exp.osp.cids;
end

spikeBinSize = 1/Exp.S.frameRate; % bin at the frame rate (4 ms bins)
gazeContingent = ip.Results.GazeContingent;

dataDir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'stim_movies');
if ~exist(dataDir, 'dir')
    mkdir(dataDir)
end

if isempty(ip.Results.correctEyePos)
    correctEyePos = 0;
    shifter = [];
else
    correctEyePos = 1;
    shifter = ip.Results.correctEyePos;
end

fname = sprintf('%s_%s_%d_%d_%d_%d_%d_%d.hdf5',...
    strrep(Exp.FileTag, '.mat', ''),...
    strrep(strrep(num2str(S.rect), ' ', '_'), '__', '_'), ... % rect
    ip.Results.GazeContingent, ...
    correctEyePos, ...
    ip.Results.includeProbe, ...
    ip.Results.eyesmooth, ...
    ip.Results.nonlinearEyeCorrection, ...
    ip.Results.usePTBdraw);

h5name = fullfile(dataDir, fname);
if ip.Results.testmode
    h5path = ['/' ip.Results.stimulus '/Test'];
else
    h5path = ['/' ip.Results.stimulus '/Train'];
end

if exist(h5name, 'file')
    
    hinfo = h5info(h5name);
    
    stimexist = (arrayfun(@(x) strcmp(x.Name(2:end), ip.Results.stimulus), hinfo.Groups));
    
    if any(stimexist)
        grp = hinfo.Groups(stimexist);
        if any(arrayfun(@(x) strcmp(x.Name, h5path), grp.Groups)) && ~ip.Results.overwrite
            fprintf('Stimulus already exported\n')
            return
        end
    end
        
end

%% Select trials to analyze
stimulusSet = ip.Results.stimulus;

fprintf('Reconstructing [%s] stimuli...\n', stimulusSet)

validTrials = io.getValidTrials(Exp, stimulusSet);
if ~isempty(ip.Results.validTrials)
    validTrials = intersect(validTrials, ip.Results.validTrials);
end

numValidTrials = numel(validTrials);

if numValidTrials == 0
    error('No valid trials')
    return
end

% --- GET EYE POSITION
eyePos = Exp.vpx.smo(:,2:3);

% --- SMOOTH EYE POSITION
if ip.Results.eyesmooth > 1 
    
    % smoothing window must be odd
    if mod(ip.Results.eyesmooth-1,2)~=0
        smwin = ip.Results.eyesmooth - 1;
    else
        smwin = ip.Results.eyesmooth;
    end
    
    eyePos(:,1) = sgolayfilt(eyePos(:,1), ip.Results.eyesmoothorder, smwin);
    eyePos(:,2) = sgolayfilt(eyePos(:,2), ip.Results.eyesmoothorder, smwin);
    
end

% % --- Correct with shifter if available
% if correctEyePos
%     iix = ~isnan(hypot(Exp.vpx.smo(:,2), Exp.vpx.smo(:,3)));
%     
% %     eyePos = io.getCorrectedEyePos(Exp, 'usebilinear', ip.Results.nonlinearEyeCorrection);
% end


% make sure we have the latency of the monitor / graphics card included
if ~isfield(S, 'Latency')
    Latency = 8.3e-3; % measurement on propixx rig in Mitchell lab
else
    Latency = S.Latency;
end

% get test index
hasFrozen = (cellfun(@(x) isfield(x.PR, 'frozenSequence'), Exp.D(validTrials)));
if any(hasFrozen) && any(cellfun(@(x) x.PR.frozenSequence, Exp.D(validTrials(hasFrozen))))
    disp('Using frozen trials as test set')
    frozenPossible = validTrials(hasFrozen);
    isFrozen = cellfun(@(x) x.PR.frozenSequence, Exp.D(validTrials(hasFrozen)));
    testTrials = frozenPossible(isFrozen);
else
    rng(666) % set random seed so the test set is always the same
    testTrials = randsample(validTrials, ceil(.1*numValidTrials)); % use 10 percent of the data
end

if ip.Results.testmode
    validTrials = sort(testTrials);
else
    validTrials = setdiff(validTrials,testTrials);
end

%% check if stimulus reconstrution has already been run
if exist(h5name, 'file')
    hinfo = h5info(h5name);
    
    [a, ~] = fileparts(h5path);
    stimid = find(arrayfun(@(x) strcmp(x.Name, a), hinfo.Groups));
    if ~isempty(stimid) % stim has been run at least once
        % check if Train/Test group has been run
        if arrayfun(@(x) strcmp(x.Name, h5path), hinfo.Groups(stimid).Groups)
            hinfo = h5info(h5name, h5path);
            
            if any(arrayfun(@(x) strcmp(x.Name, 'Stim'), hinfo.Datasets))
                disp('Already run')
                return
            end
        end
    end
end


%% reconstruct stimulus (saving online into the file)

binSize = 1; % pixel
fprintf('Regenerating stimuli...\n')

regenerateStimulus(Exp, validTrials, rect, ...
    'spatialBinSize', binSize, ...
    'GazeContingent', gazeContingent, ...
    'Shifter', shifter, ...
    'Latency', Latency, 'eyePos', eyePos, ...
    'includeProbe', ip.Results.includeProbe, ...
    'debug', ip.Results.debug, ...
    'usePTBdraw', ip.Results.usePTBdraw, ...
    'h5file', h5name, ...
    'h5path', h5path);


%% read back relevant variables for further analysis

dims = h5readatt(h5name, [h5path '/Stim'], 'size');
seedGood = h5read(h5name, [h5path '/seedGood']);
frameTimesOe = h5read(h5name, [h5path '/frameTimesOe']);
eyeAtFrame = h5read(h5name, [h5path '/eyeAtFrame']);


%% read one frame (testing)
% % start = find(seedGood,1)+1;
% start = start + 1;
% I = h5read(h5name, [h5path '/Stim'], [1, 1, start], double([dims(1:2)' 1]));
% figure(1); clf
% imagesc(I); colormap gray
% title(start)
% 

%% Get indices for analysis 

NTall = dims(3);
valdata = zeros(NTall,1);
bad = ~seedGood;
fprintf('%d frames excluded because the seed failed\n', sum(bad))
goodIx = ~bad;

bad = frameTimesOe==0; 
fprintf('%d frames excluded because the frameTime was 0\n', sum(bad(goodIx)))
goodIx = goodIx & ~bad;

valdata(goodIx) = 1;

fprintf('%d good frames at stimulus resolution (%0.2f sec)\n', sum(valdata), sum(valdata)*spikeBinSize);

% get eye pos labels
labels = Exp.vpx.Labels(eyeAtFrame(:,4));

% get saccade times
saccades = labels==2;
sstart = find(diff(saccades)==1);
sstop = find(diff(saccades)==-1);
if saccades(1)
    sstart = [1; sstart];
end

if saccades(end)
    sstop = [sstop; numel(saccades)];
end
slist = [sstart sstop];

%% Bin spike times
Y = binNeuronSpikeTimesFast(Exp.osp, frameTimesOe, spikeBinSize);
Y = Y(:,cluster_ids);

%% write to file
NC = numel(cluster_ids);

% update hdf5 file
finfo = h5info(h5name, h5path);
if ~any(arrayfun(@(x) strcmp(x.Name, 'Robs'), finfo.Datasets))
    h5create(h5name, [h5path, '/Robs'], size(Y))
end

if ~any(arrayfun(@(x) strcmp(x.Name, 'valinds'), finfo.Datasets))
    h5create(h5name, [h5path, '/valinds'], size(valdata))
end

if ~any(arrayfun(@(x) strcmp(x.Name, 'labels'), finfo.Datasets))
    h5create(h5name, [h5path, '/labels'], size(labels))
end

if ~any(arrayfun(@(x) strcmp(x.Name, 'slist'), finfo.Datasets))
    h5create(h5name, [h5path, '/slist'], size(slist))
end

h5write(h5name, [h5path, '/Robs'], Y)
h5writeatt(h5name, [h5path, '/Robs'], 'NC', NC)
h5writeatt(h5name, [h5path, '/Robs'], 'cids', cluster_ids)


h5write(h5name, [h5path, '/valinds'], valdata)


h5write(h5name, [h5path, '/labels'], labels)


h5write(h5name, [h5path, '/slist'], slist)

%%

switch ip.Results.stimulus
    case 'Grating'
        % Load grating data
        forceHartley = true;
        [~, RobsG, grating] = io.preprocess_grating_subspace_data(Exp, 'force_hartley', forceHartley, 'validTrials', validTrials);
        RobsG = RobsG(:,Exp.osp.cids);

        if numel(unique(grating.oris)) > 20 % stimulus was not run as a hartley basis
            forceHartley = false;
            [~, RobsG, grating] = io.preprocess_grating_subspace_data(Exp, 'force_hartley', forceHartley, 'validTrials', validTrials);
            RobsG = RobsG(:,Exp.osp.cids);
        end
        
        
        [~, ~, id] = histcounts(frameTimesOe, grating.frameTime);
        gframeTime = grating.frameTime(id+1);
        
        
        h5writeatt(h5name, [h5path, '/Stim'], 'oris', grating.oris)
        h5writeatt(h5name, [h5path, '/Stim'], 'cpds', grating.cpds)
        h5writeatt(h5name, [h5path, '/Stim'], 'hartley', double(grating.force_hartley))
        h5writeatt(h5name, [h5path, '/Stim'], 'frozen_start_times', grating.frameTime(grating.frozen_seq_starts))
        h5writeatt(h5name, [h5path, '/Stim'], 'frozen_frame_duration', grating.frozen_seq_dur)
        
        
        h5create(h5name, [h5path, '/grating_time'], size(gframeTime))
        h5create(h5name, [h5path, '/grating_ori'], size(gframeTime))
        h5create(h5name, [h5path, '/grating_cpd'], size(gframeTime))
        h5write(h5name, [h5path, '/grating_time'], gframeTime)
        h5write(h5name, [h5path, '/grating_ori'], grating.ori(id+1))
        h5write(h5name, [h5path, '/grating_cpd'], grating.cpd(id+1))
        
    case 'FixRsvpStim'
%       
        trial_frame_dur = cellfun(@(x) size(x.PR.NoiseHistory,1), Exp.D(validTrials));
        validTrials = validTrials(trial_frame_dur > 1);
        trial_frame_dur = trial_frame_dur(trial_frame_dur > 1);
        trial_starts = Exp.ptb2Ephys(cellfun(@(x) x.PR.NoiseHistory(1,1), Exp.D(validTrials)));
        
        
        [~, ~, id] = histcounts(trial_starts,frameTimesOe);
        h5writeatt(h5name, [h5path, '/Stim'], 'trial_starts', trial_starts)
        h5writeatt(h5name, [h5path, '/Stim'], 'trial_frame_dur', trial_frame_dur)
        h5writeatt(h5name, [h5path, '/Stim'], 'trial_frame_starts', id)
        
        
        
end


return
        
%% test loading

% h5name




% % 
% hinfo = h5info(h5name); %, h5path);
% % stimid = find(arrayfun(@(x) strcmp(x.Name(2:end), ip.Results.stimulus), hinfo.Groups));
% 
% % hinfo.Groups(stimid).Groups(2).Attributes
% 
% 
% hinfo = h5info(h5name, [h5path '/Stim']);
% hinfo.Attributes.Name




