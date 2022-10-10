function S = cumulative_fix_time(Exp)
% get the cumulative fixation time for fixation paradigms vs. free-viewing

S = struct();

S.fixtime.use = false;

if ~isempty(io.getValidTrials(Exp, 'FixRsvpStim'))
    S.fixtime.use = true;
    
    offset = 0.02; % postsaccadic offset
    
    S.fixtime.offset = offset;
    
    trstarts = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(:)));
    trstops = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(:)));
    
    sacon = Exp.vpx2ephys(Exp.slist(2:end,1));
    sacoff = Exp.vpx2ephys(Exp.slist(1:end-1,2));
    
    % --- Forage
    % get forage trial fixation time
    validTrials = io.getValidTrials(Exp, 'Forage');
    numTrials = numel(validTrials)-1;
    
    fixTimeForage = zeros(numTrials,1);
    experimentTimeForage = zeros(numTrials,1);
    
    for iTrial = 1:numTrials
        thisTrial = validTrials(iTrial);
        
        fixix = find((sacoff + offset) > trstarts(thisTrial) & sacon < trstops(thisTrial));
        
        fixTimeForage(iTrial) = sum(sacon(fixix) - (sacoff(fixix)+offset));
        experimentTimeForage(iTrial) = trstarts(thisTrial+1)-trstarts(thisTrial);
    end
    
    % --- Fixation
    % do the same for fixation
    validTrials = io.getValidTrials(Exp, 'FixRsvpStim');
    numTrials = numel(validTrials)-1;
    
    fixTimeFix = zeros(numTrials,1);
    experimentTimeFix = zeros(numTrials,1);
    
    for iTrial = 1:numTrials
        thisTrial = validTrials(iTrial);
        if isempty(Exp.D{thisTrial}.PR.NoiseHistory)
            fixTimeFix(iTrial) = 0;
        else
            fixTimeFix(iTrial) = Exp.D{thisTrial}.PR.NoiseHistory(end,1) - Exp.D{thisTrial}.PR.NoiseHistory(1,1);
        end
        experimentTimeFix(iTrial) = trstarts(thisTrial+1)-trstarts(thisTrial);
    end
    
    S.fixtime.exTimeFor = cumsum(experimentTimeForage);
    S.fixtime.fixTimeFor = cumsum(fixTimeForage);
    S.fixtime.exTimeFix = cumsum(experimentTimeFix);
    S.fixtime.fixTimeFix = cumsum(fixTimeFix);
    
    figure(1); clf
    plot(cumsum(experimentTimeForage), cumsum(fixTimeForage)); hold on
    plot(cumsum(experimentTimeFix), cumsum(fixTimeFix));
    plot(xlim, xlim, 'k')
    drawnow
end