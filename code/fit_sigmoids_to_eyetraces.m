function [fixEye, stats] = fit_sigmoids_to_eyetraces(exx, eyy, fixations, buffer, debug)
% [fixEye, stats] = fit_sigmoids_to_eyetraces(eyeX, eyeY, fixations (starts and stops), buffer, debug)
% fits saccades using sigmoids
if nargin < 5
    debug = false;
end

if nargin < 4
    buffer = 10;
end

fixX = nan*exx;
fixY = nan*eyy;

if numel(fixations) < 4
    fixX(:) = mean(exx);
    fixY(:) = mean(eyy);
    fixEye = [fixX(:) fixY(:)];
    stats = [];
    return
end

sigmoid = @(params, x) params(1) + (params(2)-params(1))./(1 + exp( (x - params(3))/params(4)));
fun = @(params, x) [sigmoid(params(1:4), x) sigmoid(params(5:end), x)];

params = [];
N = size(fixations,1)-1;
for ifix = 1:N
    if mod(ifix, 100)==0
        fprintf('%d/%d\n', ifix, N)
    end


    ix1 = (fixations(ifix,1) + buffer):(fixations(ifix,2)-buffer); % this fixation
    ix2 = (fixations(ifix+1,1) + buffer):(fixations(ifix+1,2)-buffer); % next fixation
    iix = min(ix1):max(ix2);

    % initialize x parameters
    p1x = mean(exx(ix1));
    p2x = mean(exx(ix2));
    p3x = (fixations(ifix,2) + fixations(ifix+1,1))/2;
    p4x = var(exx(iix));
    if p2x > p1x || (p2x-p1x) < 0
        p4x = -p4x;
    end
    
    % initialize y parameters
    p1y = mean(eyy(ix1));
    p2y = mean(eyy(ix2));
    p3y = p3x;
    p4y = var(eyy(iix));
    if p2y > p1y || (p2y-p1y) < 0
        p4y = -p4y;
    end

    params0x = [p1x, p2x, p3x, p4x];
    params0y = [p1y, p2y, p3y, p4y];
    
%     % fit all
%     params0 = [params0x params0y];
%     evalc("paramsHat = lsqcurvefit(fun, params0, iix', [exx(iix) eyy(iix)])");
%     params = [params; paramsHat];

    % don't fit fixations
    fun = @(params, x) [sigmoid([p1x, params(1:3)], x), sigmoid([p1y, params(4:6)], x)];
    params0 = [params0x(2:4) params0y(2:4)];
    evalc("paramsHat = lsqcurvefit(fun, params0, iix', [exx(iix) eyy(iix)])");
    params = [params; [p1x paramsHat(1:3) p1y paramsHat(4:6)]];

    if debug
        figure(1); clf
        subplot(1,2,1)
        plot(iix, exx(iix), 'k')
        hold on
%         plot(iix, sigmoid(params0x, iix))
        plot(iix, sigmoid(params(end,1:4), iix))
        subplot(1,2,2)
        plot(iix, eyy(iix), 'k')
        hold on
        plot(iix, sigmoid(params(end,5:end), iix))

        pause
    end

    fixX(iix) = sigmoid(params(end,1:4), iix);
    fixY(iix) = sigmoid(params(end,5:end), iix);
    
end


fixEye = [fixX fixY];
fun = @(params, x) [sigmoid(params(1:4), x) sigmoid(params(5:end), x)];
stats = struct('params', params, ...
    'fun', fun, 'sigmoid', sigmoid);
