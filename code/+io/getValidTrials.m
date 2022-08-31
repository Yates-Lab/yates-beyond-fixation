function validTrials = getValidTrials(Exp, stimulusSet)
% validTrials = getValidTrials(Exp, stimulusSet)

if nargin < 1
    fprintf('Recognized stimulus sets are:\n')
    stimList = {'Grating', 'Gabor', 'Dots', 'BackImage', ...
        'Forage', ...
        'FixRsvpStim', ...
        'FixCalib', ...
        'ForageStaticLines', ...
        'FixFlashGabor', ...
        'MTDotMapping', ...
        'DriftingGrating'};
    fprintf('%s\n', stimList{:})
    validTrials = [];
    return
end
        
trialProtocols = cellfun(@(x) x.PR.name, Exp.D, 'uni', 0);
if nargin < 2
    stims = unique(trialProtocols);
    fprintf('%s\n', stims{:})
    validTrials = [];
    return
end

% --- find the trials that we want to analyze
ephysTrials = find(cellfun(@(x) ~isnan(x.START_EPHYS) | ~isnan(x.END_EPHYS), Exp.D));
if (numel(ephysTrials)/numel(Exp.D)) < 0.6
    disp('Something is wrong. Assuming all trials had ephys')
    ephysTrials = 1:numel(Exp.D);
end


switch stimulusSet
    case {'Grating'}
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==1, Exp.D(validTrials)));
        
    case {'Gabor'}
        % Forage trials
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==4, Exp.D(validTrials)));
        
    case {'BigDots'}
        % Forage trials
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

        % dot spatial noise trials
        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==5, Exp.D(validTrials)));

        dotSize = cellfun(@(x) x.P.dotSize, Exp.D(validTrials));
        validTrials = validTrials(dotSize==max(dotSize));
        
    case {'Dots'}
        % Forage trials
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

        % dot spatial noise trials
        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==5, Exp.D(validTrials)));

        dotSize = cellfun(@(x) x.P.dotSize, Exp.D(validTrials));
        validTrials = validTrials(dotSize==min(dotSize));
        
    case {'CSD'}
        % Forage trials
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

        % dot spatial noise trials
        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==3, Exp.D(validTrials)));
        
    case {'BackImage'}
        validTrials = intersect(find(strcmp(trialProtocols, 'BackImage')), ephysTrials);
    
    case {'DriftingGrating'}
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);

        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==6, Exp.D(validTrials)));
        
    case {'Forage'}
        
        validTrials = intersect(find(strcmp(trialProtocols, 'ForageProceduralNoise')), ephysTrials);
    
    case {'FixFlash'}
        
        validTrials = find(strcmp(trialProtocols, 'FixFlash'));
        
    case {'FaceCal'}
        
        validTrials = find(strcmp(trialProtocols, 'FaceCal'));
    
    case {'FixCalib'}
        
        validTrials = find(strcmp(trialProtocols, 'FixCalib'));
        
    case {'MTDotMapping'}
        
        validTrials = intersect(find(strcmp(trialProtocols, 'Forage')), ephysTrials);
        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==5, Exp.D(validTrials)));
    
    case {'FixFlashGabor'}
        validTrials = intersect(find(strcmp(trialProtocols, 'FixFlash_ProceduralNoise')), ephysTrials);
        validTrials = validTrials(cellfun(@(x) x.PR.noisetype==4, Exp.D(validTrials)));
        
    case {'All'}
        % use all valid conditions (BackImage or
        validTrials = find(strcmp(trialProtocols, 'BackImage') | strcmp(trialProtocols, 'ForageProceduralNoise'));
        validTrials = intersect(validTrials, ephysTrials);
    case {'Ephys'}
        validTrials = find(cellfun(@(x) ~isnan(x.START_EPHYS) | ~isnan(x.END_EPHYS), Exp.D));
    otherwise
        % use all valid conditions (BackImage or
        validTrials = find(strcmp(trialProtocols, stimulusSet));
        validTrials = intersect(validTrials, ephysTrials);
        
end


numValidTrials = numel(validTrials);
fprintf('Found %d valid trials\n', numValidTrials)