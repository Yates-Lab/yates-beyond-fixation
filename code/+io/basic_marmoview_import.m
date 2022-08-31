function Exp = basic_marmoview_import(DataFolder, varargin)

ip = inputParser();
ip.addParameter('Protocols', [])
ip.addParameter('TAGSTART', 63)
ip.addParameter('TAGEND', 62)
ip.addParameter('EphysFreq', 30e3)
ip.addParameter('Timestamps', [])
ip.addParameter('fid', 1) % default is to dump to the command window
ip.parse(varargin{:});

ProtocolList = ip.Results.Protocols;
TAGSTART = ip.Results.TAGSTART;
TAGEND = ip.Results.TAGEND;

if ~isempty(ProtocolList)
    error('basic_marmoview_import: Protocol list should allow you to select the protocols you want, but it is not implemented yet')
end

fid = ip.Results.fid; % fprintf target (1 = command window)

%% Events File importing
%******** now grab the events file with strobes
HASEPHYS = false;

if ~isempty(ip.Results.Timestamps)
    HASEPHYS = true;
    tstrobes = ip.Results.Timestamps.tstrobes;
    strobes = ip.Results.Timestamps.strobes;
    fprintf('Using Timestamps struct to synch ephys\n')
else


    EventFiles = dir([DataFolder,filesep,'*.events']);
    ext = 'events';
    HASEPHYS = true;

    if isempty(EventFiles)
        EventFiles = dir(fullfile(DataFolder,'*.kwe'));
        if ~isempty(EventFiles)
            ext = 'kwe';
            HASEPHYS = true;
        else
            fprintf(fid, 'basic_marmoview_import: No ephys found\n');
        end
    end


    if HASEPHYS
        switch ext
            case 'events'
                fprintf(fid, 'Reading strobe times from [%s] .events file\n', EventFiles(1).name);
                [evdata,evtime,evinfo] = read_ephys.load_open_ephys_data_faster([DataFolder,filesep,EventFiles(1).name]);
            case 'kwe'
                fprintf(fid, 'Reading strobe times from [%s] .kwe file\n', EventFiles(1).name);
                [evdata,evtime,evinfo] = read_ephys.load_kwe(fullfile(DataFolder,EventFiles(1).name));
                evtime = double(evtime) / ip.Results.EphysFreq; % convert to seconds
        end
        %**** convert events into strobes with times
        [tstrobes,strobes] = read_ephys.convert_data_to_strobes(evdata,evtime,evinfo);
        strobes = read_ephys.fix_missing_strobes(strobes);
        fprintf(fid, 'Strobes are loaded\n');

    end

    if ~HASEPHYS
        % check for "Digital In" files from an intan system
        diginFile = dir(fullfile(DataFolder, '*DigIn.mat'));
        if ~isempty(diginFile)
            disp("Found INTAN strobe file")
            dig = load(fullfile(DataFolder, diginFile(1).name));

            % This is a hacky piece of code to recover the strobes from the
            % intan system
            % The state of the Digital INs is sampled at the full sampling rate
            % of the system (30kHz) meaning we get many samples with the same
            % bit on and we have to find the set and unset times after the
            % fact.
            strobe_id = find(strcmp('DIGITAL-IN-07', {dig.board_dig_in_channels(:).native_channel_name}));
            tstrobes = find(diff(dig.board_dig_in_data(strobe_id,:))>0)+1; % find set times
            nbitflips = numel(tstrobes);
            strobes = zeros(nbitflips, 1);
            for i = 1:nbitflips
                ind = tstrobes(i);
                strobe_ = dig.board_dig_in_data(:,ind)';
                strobes(i) = bin2dec(num2str( fliplr(strobe_(1:6)) ));
            end

            tstrobes = tstrobes / ip.Results.EphysFreq; % convert to seconds
            HASEPHYS = true;
        end


    end
end

%% Loading up the MarmoView Data Files
%****************************************************************
ExpFiles = dir([DataFolder,filesep,'*z.mat']);
if isempty(ExpFiles)
    error('basic_marmoview_import: Error finding *z.mat file');
end

%****** get order by date of files to import them
FileDates = cellfun(@datenum,{ExpFiles(:).date});
DoFileSort = [ (1:size(FileDates,2))' FileDates'];
FileSort = sortrows(DoFileSort,2); % sort them by date
%***** read and append files in order by date *********
BigN = 0;
for zk = FileSort(:,1)'
    fname = ExpFiles(zk).name;
    load([DataFolder,filesep,fname]);
    if ~BigN
        Exp.D = D;
        Exp.S = S;
        BigN = size(D,1); % number of trials
    else
        for k = 1:size(D,1)
            Exp.D{BigN+k} = D{k};  % appending trials
        end
        BigN = BigN + size(D,1);
    end
    clear D;
    clear S;
    fprintf(fid, 'Experiment file %s loaded\n',fname);
end
%***** store spikes info in Exp struct, let's keep all info there
%***** once finished we will clear everything but the Exp struct
if exist('osp', 'var')
    Exp.osp = osp;
end
if exist('sp', 'var')
    Exp.sp = sp;
else
    Exp.sp = [];
end


clear sp lfp
fprintf(fid, 'All experiment files loaded\n');
%**************************

%% Synching up strobes from Ephys to MarmoView
%******************* returns start and end times in ephys record, or NaN if missing
%***** this code would look simple, but sometimes one bit is flipped in
%***** error and then you have to play catch up to find the missing code
%***** since there is some redundancy (start and end codes for each trial)
%***** this gives us a way to recover cases with just one errant bit
if HASEPHYS
    fprintf(fid, '\nSynching up ephys strobes\n\n');
    for k = 1:size(Exp.D,1)
        start = synchtime.find_strobe_time(TAGSTART,Exp.D{k}.STARTCLOCK,strobes,tstrobes);
        finish = synchtime.find_strobe_time(TAGEND,Exp.D{k}.ENDCLOCK,strobes,tstrobes);
        if (isnan(start) || isnan(finish))
            fprintf(fid, 'Synching trial %d\n',k);
            if (isnan(start) && isnan(finish))  % if both are missing drop the trial
                fprintf(fid, 'Dropping entire trial %d from protocol %s\n',k,Exp.D{k}.PR.name);
                Exp.D{k}.START_EPHYS = NaN;
                Exp.D{k}.END_EPHYS = NaN;
            else
                %******* here we could try to repair a missing code, or find a
                %******* a partial code nearby
                Exp.D{k}.START_EPHYS = start;
                Exp.D{k}.END_EPHYS = finish;
                tdiff = Exp.D{k}.eyeData(end,6) - Exp.D{k}.eyeData(1,1);
                if isnan(start) && ~isnan(finish)
                    fprintf(fid, '**Approximating start code from end\n');
                    Exp.D{k}.START_EPHYS = Exp.D{k}.END_EPHYS - tdiff;
                    %****** now see if you can do even better
                    %****** see if the real code is there but a bit flipped
                    zz = find(tstrobes == finish);  % find end code
                    istart = zz(1) - 7;
                    if (istart >= 1) && (strobes(istart) == TAGSTART)  % candidate start before end
                        beftag = strobes((istart+1):(istart+6))';
                        mato = sum( Exp.D{k}.STARTCLOCK & beftag );
                        if (mato >= 5)  % all but one of taglet matched
                            E.D{k}.START_EPHYS = tstrobes(istart);
                            fprintf(fid, '****Located matching start strobe, one bit was flipped\n');
                        end
                    end
                    %*******************************
                end
                if isnan(finish) && ~isnan(start)
                    fprintf(fid, '##Approximating end code from start\n');
                    Exp.D{k}.END_EPHYS = Exp.D{k}.START_EPHYS + tdiff;
                    %****** now see if you can do even better
                    %****** see if the real code is there but a bit flipped
                    zz = find(tstrobes == start);  % find end code
                    istart = zz(1) + 7;
                    if (istart < size(strobes,1)) && (strobes(istart) == TAGEND)  % candidate start before end
                        if numel(strobes) >= (istart + 6)
                            endtag = strobes((istart+1):(istart+6))';
                            mato = sum( Exp.D{k}.ENDCLOCK & endtag );
                            if (mato >= 5)  % all but one of taglet matched
                                E.D{k}.END_EPHYS = tstrobes(istart);
                                fprintf(fid, 'ENDTAG: %d\n', endtag);
                                fprintf(fid, '####Located matching end strobe, one bit was flipped\n');
                            end
                        end
                    end
                    %*******************************
                end
                %****************
            end
        else
            Exp.D{k}.START_EPHYS = start;  % otherwise store the NaN -- maybe trial not used
            Exp.D{k}.END_EPHYS = finish;
        end
    end
    
    
    if all(cellfun(@(x) isnan(x.START_EPHYS), Exp.D))
        fprintf(fid, 'Syncing failed catastrophically. Trying to force it...\n');
        
        sixletsOE = fliplr(conv2(strobes, eye(6)));
        sixletsOE=sixletsOE(6:end,:);
        
        numTrials = numel(Exp.D);
        for kTrial = 1:numTrials
            startId = find(all(Exp.D{kTrial}.STARTCLOCK == sixletsOE,2));
            endId = find(all(Exp.D{kTrial}.ENDCLOCK == sixletsOE,2));
            if ~isempty(startId) && numel(startId)==1
                fprintf(fid, 'Found STARTCLOCK for trial %d\n', kTrial);
                Exp.D{kTrial}.START_EPHYS = tstrobes(startId);
            else
                Exp.D{kTrial}.START_EPHYS = nan;
            end
            
            if ~isempty(endId) && numel(endId)==1
                fprintf(fid, 'Found ENDCLOCK for trial %d\n', kTrial);
                Exp.D{kTrial}.END_EPHYS = tstrobes(endId);
            else
                Exp.D{kTrial}.END_EPHYS = nan;
            end
        end
        
        
    end
    
    % do a global synchronization of the clocks
    Exp.ptb2Ephys = synchtime.sync_ptb_to_ephys_clock(Exp, [], 'fid', fid);
    fprintf(fid, 'Finished Synching up ephys strobes\n');
    
    
    
else
    
    numTrials = numel(Exp.D);
    for iTrial = 1:numTrials
        Exp.D{iTrial}.START_EPHYS = nan;
        Exp.D{iTrial}.END_EPHYS = nan;
    end
    Exp.ptb2Ephys = @(x) x; % do nothing
    
end