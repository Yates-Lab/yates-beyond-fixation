function S = grating_RF_single_session(Exp, varargin)
% get receptive field info for grating stimuli

ip = inputParser();
ip.addParameter('numlags', 15)
ip.addParameter('plot', true)
ip.parse(varargin{:})

%% initialize
NC = numel(Exp.osp.cids);
rfdeets = repmat(struct('sta', nan), NC, 1); % initialize struct
for cc = 1:NC
    rfdeets(cc).sta = nan;
    rfdeets(cc).rffit.mu = nan(1,2);
    rfdeets(cc).rffit.cov = nan(2);
    rfdeets(cc).rffit.base = nan;
    rfdeets(cc).rffit.amp = nan;
    rfdeets(cc).peaklag = nan;
    rfdeets(cc).srf = nan;
    rfdeets(cc).xax = nan;
    rfdeets(cc).yax = nan;
    rfdeets(cc).cid = cc;
end

forceHartley = true;
[Stim, Robs, grating] = io.preprocess_grating_subspace_data(Exp, 'force_hartley', true);
Robs = Robs(:,Exp.osp.cids);

if numel(unique(grating.oris)) > 20 % stimulus was not run as a hartley basis
    forceHartley = false;
    [Stim, Robs, grating] = io.preprocess_grating_subspace_data(Exp, 'force_hartley', false);
    Robs = Robs(:,Exp.osp.cids);
end

%% embed time
nlags = ip.Results.numlags;
Xstim = makeStimRows(Stim{1}, nlags);

% find valid fixations
valid = find(grating.eyeLabel==1);

nValid = numel(valid);
fprintf('%d valid (fixation) samples\n', nValid)

%% estimate receptive field
% assume smoothness in space and time.
% estimate the amount of smoothness by maximizing the model evidence
% using a gridsearch

% compute Response-Triggered Average (in units of delta spike rate)
Rvalid = Robs - mean(Robs);

%   train = valid(1:floor(4*nValid/5))';
%   test = valid(ceil(4*nValid/5):nValid)';
% random sample train and test set
test = randsample(valid, floor(nValid/5));
train = setdiff(valid, test);

XX = (Xstim(train,:)'*Xstim(train,:)); % covariance
XY = (Xstim(train,:)'*Rvalid(train,:));

%%

NC = size(Rvalid,2);

for cc = 1:NC
    
    Rtest = Rvalid(test,:);
    Rtest = imgaussfilt(Rtest(:,cc), 2);
    
    CpriorInv = qfsmooth3D([nlags, fliplr(grating.dim)], [.5 .5]);
    CpriorInv = CpriorInv + eye(size(CpriorInv,2));
    
    % if you want to check that you're in the right range
    
    
%     if ip.Results.plot
%         % initialize
%         lambda0 = 1e3;
%         kmap0 = (XX+lambda0*CpriorInv)\XY(:,cc);% ./sum(Xstim(valid,:))';
%         
%         rfspt = reshape(kmap0, nlags, []);
%         
%         
%         [~, peak_lag] = max(std(rfspt,[],2));
%         figure(2);
%         clf
%         
%         subplot(1,2,1)
%         plot(rfspt)
%         subplot(1,2,2)
%         I = reshape(rfspt(peak_lag,:), (grating.dim));
%         I = [rot90(I,2) I]; %#ok<*AGROW> % mirror Fourier domain
%         imagesc(I)
%         title(cc)
%     end
    
    
    % loop over hyperparameter
    lambdas = (2.^(-1:15));
    
    r2test = zeros(numel(lambdas),1);
    
    for il = 1:numel(lambdas)
        
        % get posterior weights
        lambda = lambdas(il);
        
        H = XX + lambda*CpriorInv;
        
        kmap = H\XY(:,cc);  % Posterior Mean (MAP estimate)
        
        % predict on test set
        Rhat = Xstim(test,:)*kmap;
        
        
        r2test(il) = rsquared(Rtest, Rhat);
        
    end
    
    if ip.Results.plot
        figure(1); clf
        plot(lambdas, r2test)
    end
    
    [r2test, id] = max(r2test);
    lambda = lambdas(id);
    
    
    H = XX + lambda*CpriorInv;
    kmap = H\XY(:,cc);
    
    sta = kmap;
    
    rfspt = reshape(sta, nlags, []);
    
    rfdeets(cc).sta = rfspt;
    rfdeets(cc).r2test = r2test;
    
    % get temporal kernel (assuming separability)
    [u,~,~] = svd(rfspt);
    tk = u(:,1) ./ sum(u(:,1));
    
    % find peak lag
    [~, peaklag] = max(tk);
    
    %% reshape dimensions
    I = reshape(rfspt(peaklag,:), grating.dim);
    I = I ./ max(I(:));
    I = I';
    
    % --- fit parametric RF    
    [y0,x0] = find(I==1, 1);

    x0 = grating.oris(x0);
    y0 = grating.cpds(y0);
    
%     figure(3); clf
%     imagesc(grating.oris, grating.cpds, I); hold on
%     plot(x0, y0, 'or')
    
    if forceHartley
        [theta0, cpd0] = cart2pol(x0, y0);
    else
        theta0 = x0/180*pi;
        cpd0 = y0;
    end
    
    par0 = [1 0 cpd0 1 max(I(:)) 0];
    
%     figure(1); clf
%     plot(I(:)); hold on
%     plot(prf.parametric_rf(par0, X)); 
    
    mxI = max(I(:));
    mnI = min(I(:));
    
    % least-squares for parametric rf
    fun = @(params) (prf.parametric_rf([params mxI mnI], X) - I(:)).^2;
    

    if forceHartley % space is hartley
        [xx,yy] = meshgrid(grating.oris, grating.cpds);
        [X(:,1), X(:,2)] = cart2pol(xx(:), yy(:));
    else
        [xx,yy] = meshgrid(grating.oris/180*pi, grating.cpds);
        X = [xx(:) yy(:)];
    end
    
    par0 = [2 theta0 cpd0 1]; % initial parameters
    
%     Ihat = prf.parametric_rf([par0 max(I(:)) min(I(:))], X);
%     figure(1); clf
%     subplot(1,2,1)
%     plot(I(:)); hold on
%     plot(Ihat(:)); 
%     subplot(1,2,2)
%     imagesc(reshape(Ihat, size(xx)))
%     
%     evalc('phat = lsqnonlin(fun, par0)');
%     
%     Ihat = prf.parametric_rf([phat max(I(:)) min(I(:))], X);
%     subplot(1,2,1)
%     plot(Ihat)
    
    
    
%     figure(1); clf
%     contourf(xx,yy, I); hold on
%     plot(theta0, cpd0, 'or')
    
    % least-squares
    try
        %  parameters are:
        %   1. Orientation Kappa
        %   2. Orientation Preference
        %   3. Spatial Frequency Preference
        %   4. Spatial Frequency Sigma
        %   5. Gain
        %   6. Offset
        lb = [.01 -pi 0.1 .2];
        ub = [10 2*pi 20 1];
        
        evalc('phat = lsqnonlin(fun, par0, lb, ub)');
%         evalc('phat = lsqcurvefit(fun, par0, X, I(:), lb, ub);');
    catch
        phat = nan(size(par0));
    end
    
    phat = [phat mxI mnI];
%     Ihat = reshape(fun(phat), size(xx));
%     Ihat0 = reshape(fun(par0), size(xx));
    Ihat = reshape(prf.parametric_rf(phat, X), size(xx));
    Ihat0 = reshape(prf.parametric_rf([par0 phat(5:6)], X), size(xx));
    
    
%     Ihat = reshape(fun(phat, X), size(xx));
%     Ihat0 = reshape(fun(par0, X), size(xx));
    
%     if rsquared(I(:), Ihat(:)) < .1
%         phat = par0;
%         Ihat = reshape(fun(phat, X), size(xx));
%     end
        
    if ip.Results.plot
        figure(2); clf
        
        subplot(2,2,2)
        imagesc(I)
        title('RF')
        
        subplot(2,2,4)
        imagesc(Ihat)
        title('Fit')
        
        subplot(2,2,3)
        
        imagesc(Ihat0)
        title('Fit0')
        
        plot(I(:)); hold on
        plot(Ihat(:))
        plot(Ihat0(:))
        
        
%         rsquared(I(:), Ihat0(:))
        
        subplot(2,2,1)
        plot(rfspt, 'k')
        thresh = sqrt(robustcov(sta(:)))*4;
        hold on
        title('Full STA')
        plot(xlim, thresh*[1 1])
    end


    
    %% save results
    rfdeets(cc).rffit.oriPref = phat(2)/pi*180;
    rfdeets(cc).rffit.oriBandwidth = phat(1)/pi*180;
    rfdeets(cc).rffit.sfPref = phat(3);
    rfdeets(cc).rffit.sfBandwidth = phat(4);
    rfdeets(cc).rffit.base = phat(6);
    rfdeets(cc).rffit.amp = phat(5);
%     ghat = fun(phat, X);
    ghat = prf.parametric_rf(phat, X);
    rfdeets(cc).rffit.r2 = rsquared(I(:), ghat(:));
    rfdeets(cc).peaklag = peaklag;
    rfdeets(cc).srf = I;
    rfdeets(cc).cpds = grating.cpds;
    rfdeets(cc).oris = grating.oris;
    rfdeets(cc).cid = cc;
end

S = rfdeets;