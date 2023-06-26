

classdef calibGUI < handle
    % GUI for fixing calibration
    %
    % Purpose is to do the manual calibration thing offline before
    % refinement
    %
    % GUI by J.L. Yates
    
    properties (Access = public)
        H % handles
        Exp % Exp struct
        xy % eye position
        spd % eye speed
        cmat = [1 1 0 0 0 ] % [scale x, scale y, rotation, offset x, offset y]
        loss
        id
        validix
        targets
        trialTargets
        raw_field
        cmap
        cmatHat = []
        xlim = [-12 12]
        ylim = [-12 12]
        plotid
        datasetFile = []
        selected = []
        gaussian_sigma = 10
    end
    
    methods (Access = public)
        function obj = calibGUI(Exp, varargin)
            
            ip = inputParser();
            ip.addParameter('datasetFile', [])
            ip.addParameter('use_smo', false)
            ip.addParameter('sigma', 10)
            ip.parse(varargin{:});
            
            
            fig = figure(1029321); clf % ks uses some defined figure numbers for plotting - with this random number we don't clash
            set(fig,'Name', 'calibGUI',...
                'MenuBar', 'none',...
                'Toolbar', 'none',...
                'NumberTitle', 'off',...
                'Units', 'normalized',...
                'OuterPosition', [0.1 0.1 0.8 0.8]);
            
            
            % check that required functions are present
            if ~exist('uiextras.HBox') %#ok<EXIST>
                error('sacGUI:init:uix', 'You must have the "uiextras" toolbox to use this GUI. Choose Home->Add-Ons->Get Add-ons and search for "GUI Layout Toolbox" by David Sampson. You may have to search for the author''s name to find the right one for some reason. If you cannot find it, go here to download: https://www.mathworks.com/matlabcentral/fileexchange/47982-gui-layout-toolbox\n')
            end
            obj.raw_field = 'raw';
            if ip.Results.use_smo
                obj.raw_field = 'smo';
            end

            obj.Exp = Exp;
            obj.build(fig);
            obj.initialize();
            obj.gaussian_sigma = ip.Results.sigma;
            obj.datasetFile = ip.Results.datasetFile;
            
        end
        
        function initialize(obj)
            
            validTrials = io.getValidTrials(obj.Exp, 'FaceCal');
            
            tstart = obj.Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, obj.Exp.D(validTrials)));
            tstop = obj.Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, obj.Exp.D(validTrials)));
            
            
            eyeTime = obj.Exp.vpx2ephys(obj.Exp.vpx.(obj.raw_field)(:,1));
            obj.validix = getTimeIdx(eyeTime, tstart, tstop) & ~isnan(sum(obj.Exp.vpx.(obj.raw_field)(:,2:3),2));
            
            
            obj.xy = obj.Exp.vpx.(obj.raw_field)(obj.validix,2:3);
            
            obj.spd = abs(obj.Exp.vpx.smo(obj.validix,7));
            
            obj.trialTargets = cellfun(@(x) x.PR.faceconfig(:,1:2), obj.Exp.D(validTrials), 'uni', 0);
            obj.targets = unique(cell2mat(obj.trialTargets),'rows');
            
            plot_calibration(obj)
            
            % -- initialize calibration
            n = sum(obj.validix);
            ix = true(n,1);
%             ix = all(abs(zscore(obj.xy(ix,:)))<1,2); % remove outliers
            ix = ix & ( obj.spd / nanmedian(obj.spd) < 5); % find fixations
            
            % bin on grid
            [C, xax, yax] = histcounts2(obj.xy(ix,1), obj.xy(ix,2), 1000);
            [xx,yy] = meshgrid(xax(1:end-1), yax(1:end-1));
            
            % smooth binning (like a kernel density estimate)
            I =  imgaussfilt(C', obj.gaussian_sigma);

            % softmax to find center
            wts = I(:).^9 / sum(I(:).^9);
            x = xx(:)'*wts;
            y = yy(:)'*wts;
            ctr = [x,y]; % center of mass of measured eye position
            
            % loss for aligning eye position to grid
            lossfun = @(params) sum(calibration_loss([params ctr], obj.xy(ix,:), obj.targets));

            % fit model
            opts = optimset('Display', 'none');
            phat = fminsearch(lossfun, [std(obj.xy)/10 0], opts);
            phat = [phat ctr];
            
            obj.cmat = phat;

            obj.get_eyepos(); % update eye position
            
            plot_calibration(obj)
             
        end
        
         function build(obj, f)
            % construct the GUI with appropriate panels
            obj.H.fig = f;
            set(f, 'UserData', obj);
            set(f, 'KeyPressFcn', @(f,k)obj.keyboardFcn(f, k));
            
            % plot
            obj.H.xpos = axes('Position', [.1 .2 .5 .7]);
            
            % --- individual buttons
            obj.H.refine_calibration = uicontrol( 'Parent', obj.H.fig, ...
                    'String', 'Refine Calibration', ...
                    'Units', 'normalized', ...
                    'Position', [.65 .7 .1 .05], ...
                    'Callback', @(~,~)obj.refine_calibration);
            
            obj.H.update_cmat = uicontrol( 'Parent', obj.H.fig, ...
                    'String', 'Update Calib Matrix', ...
                    'Units', 'normalized', ...
                    'Position', [.65 .65 .1 .05], ...
                    'Callback', @(~,~)obj.update_cmat);
                
            obj.H.update_plot = uicontrol( 'Parent', obj.H.fig, ...
                    'String', 'Update Plot', ...
                    'Units', 'normalized', ...
                    'Position', [.65 .6 .1 .05], ...
                    'Callback', @(~,~)obj.plot_calibration);
                
            obj.H.save = uicontrol( 'Parent', obj.H.fig, ...
                    'String', 'Save To File', ...
                    'Units', 'normalized', ...
                    'Position', [.85 .6 .1 .05], ...
                    'Callback', @(~,~)obj.saveFile);
                
            obj.H.save = uicontrol( 'Parent', obj.H.fig, ...
                    'String', 'Flip X', ...
                    'Units', 'normalized', ...
                    'Position', [.75 .7 .1 .05], ...
                    'Callback', @(~,~)obj.flipX);
                
            obj.H.save = uicontrol( 'Parent', obj.H.fig, ...
                    'String', 'Flip Y', ...
                    'Units', 'normalized', ...
                    'Position', [.75 .65 .1 .05], ...
                    'Callback', @(~,~)obj.flipY);
                
            obj.H.save = uicontrol( 'Parent', obj.H.fig, ...
                    'String', 'Show Asym Trials', ...
                    'Units', 'normalized', ...
                    'Position', [.75 .6 .1 .05], ...
                    'Callback', @(~,~)obj.show_asymetric_trials);
            
            obj.H.text.Buttons = annotation('textbox', [.67 .75 .1 .05], 'String', 'Buttons', 'Linestyle', 'none', 'Fontsize', 18);
            
            obj.H.text.Controls = annotation('textbox', [.7 .4 .2 .05], 'String', 'Keyboard Controls', 'Linestyle', 'none', 'Fontsize', 18);
            
            obj.H.text.shift = annotation('textbox', [.65 .35 .2 .05], 'String', 'Shift:     Use arrows', 'Linestyle', 'none', 'Fontsize', 18);
            obj.H.text.rotate = annotation('textbox', [.65 .3 .2 .05], 'String',  'Rotate:    < (left) > (right)', 'Linestyle', 'none', 'Fontsize', 18);
            obj.H.text.gain.head = annotation('textbox', [.65 .25 .2 .05], 'String',  'Gain:', 'Linestyle', 'none', 'Fontsize', 18);
            obj.H.text.gain.righta = annotation('arrow', [0.68 .7], [.2 .2]+.03);
            obj.H.text.gain.right = annotation('textbox', [.7 .2 .2 .05], 'String',  "Key: '", 'Linestyle', 'none', 'Fontsize', 18);
            obj.H.text.gain.lefta = annotation('arrow', [0.7 .68], [.15 .15]+.03);
            obj.H.text.gain.left = annotation('textbox', [.7 .15 .2 .05], 'String',  "Key: L", 'Linestyle', 'none', 'Fontsize', 18);
            obj.H.text.gain.upa = annotation('arrow', [0.69 .69], [.1 .13]+.02);
            obj.H.text.gain.up = annotation('textbox', [.7 .1 .2 .05], 'String',  "Key: P", 'Linestyle', 'none', 'Fontsize', 18);
            obj.H.text.gain.upa = annotation('arrow', [0.69 .69], [.08 .05]+.02);
            obj.H.text.gain.up = annotation('textbox', [.7 .05 .2 .05], 'String',  "Key: ;", 'Linestyle', 'none', 'Fontsize', 18);
            
%             obj.H.Plots = uiextras.HBox('Parent', f,...
%                 'DeleteFcn', @(~,~)obj.cleanup(), 'Visible', 'on', ...
%                 'Spacing', 0, 'Padding', 0);
            
            
            % --- plotting axes
%             obj.H.xpos = axes( 'Parent', obj.H.Plots );
%             obj.H.ypos = axes( 'Parent', obj.H.Plots );
            
            % button box
%             obj.H.Buttons = uiextras.VBox( 'Parent', obj.H.Plots);
%             obj.H.Controls = uiextras.BoxPanel( 'Parent', obj.H.Buttons, 'Title', 'Calibration Controls');
%             obj.H.ButtonBox = uiextras.VButtonBox( 'Parent', obj.H.Controls);
%             
%             % --- individual buttons
%             obj.H.refine_calibration = uicontrol( 'Parent', obj.H.ButtonBox, ...
%                     'String', 'Refine Calibration', ...
%                     'Callback', @(~,~)obj.refine_calibration);
%             
%             obj.H.update_cmat = uicontrol( 'Parent', obj.H.ButtonBox, ...
%                     'String', 'Update Calib Matrix', ...
%                     'Callback', @(~,~)obj.update_cmat);
%                 
%             obj.H.update_plot = uicontrol( 'Parent', obj.H.ButtonBox, ...
%                     'String', 'Update Plot', ...
%                     'Callback', @(~,~)obj.plot_calibration);
%             
%             set(obj.H.ButtonBox, 'Widths', [10; 10; 10])  
            
            title(obj.H.xpos, 'Loss')
            xlabel(obj.H.xpos, 'Degrees')
            ylabel(obj.H.xpos, 'Degrees')
            

            % --- Data traces initialize
            obj.H.hx = [];
            obj.H.hy = [];
            obj.H.selected = [];
            
        end
        
        
        function plot_calibration(obj)
            
            ntargs = size(obj.targets,1);
            if isempty(obj.cmap)
                obj.cmap = jet(ntargs);
            end
            
            obj.spd = abs(obj.Exp.vpx.smo(obj.validix,7));
            
            n = sum(obj.validix);
            ix = true(n,1);
%             ix = all(abs(zscore(obj.xy(ix,:)))<1,2); % remove outliers
            ix = ix & ( obj.spd / nanmedian(obj.spd) < 5); % find fixations
            
            % check loss
            [obj.loss, obj.id] = calibration_loss([1 1 0 0 0], obj.xy(ix,:), obj.targets);
            
            hold(obj.H.xpos, 'off')
            
            [C, xax, yax] = histcounts2(obj.xy(ix,1), obj.xy(ix,2), 1000);
            
            % smooth binning (like a kernel density estimate)
            I =  imgaussfilt(C', obj.gaussian_sigma);
            
            
            imagesc(obj.H.xpos, xax, yax, sqrt(I)); 
            hold(obj.H.xpos, 'on')
            
%             inds = find(ix);    
            for j = 1:ntargs

                
                obj.H.hy(j) = plot(obj.H.xpos, obj.targets(j,1), obj.targets(j,2), 'ok', 'MarkerFaceColor', obj.cmap(j,:), 'Linewidth', 2);
                    
%                 hold(obj.H.xpos, 'on')
%                 
%                 if sum(obj.id==j)==0
%                     continue
%                 end
%                 
%                 obj.H.hx(j) = plot(obj.H.xpos, obj.xy(inds(obj.id==j),1), obj.xy(inds(obj.id==j),2), '.', 'Color', obj.cmap(j,:));
%                  
                
            end      
            
            obj.H.xpos.XLim = obj.xlim;
            obj.H.xpos.YLim = obj.ylim;
            
            
            obj.H.xpos.Title.String = sprintf('Loss: %02.5f', sum(obj.loss));
            
            
        end
        
        function show_asymetric_trials(obj)
           
            %%
            
            validTrials = io.getValidTrials(obj.Exp, 'FaceCal');
            
            tstart = obj.Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, obj.Exp.D(validTrials)));
            tstop = obj.Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, obj.Exp.D(validTrials)));
            
            eyeTime = obj.Exp.vpx2ephys(obj.Exp.vpx.(obj.raw_field)(:,1));
            eyePos = obj.get_eyepos();
            
            mx = cellfun(@(x) mean(x(:,1)), obj.trialTargets);
            my = cellfun(@(x) mean(x(:,2)), obj.trialTargets);
            asymTrials = find(mx ~= 0 | my ~=0);
            nAsym = numel(asymTrials);
            figure(100); clf
            sx = ceil(sqrt(nAsym));
            sy = round(sqrt(nAsym));
            
            for iTrial = 1:nAsym
                thisTrial = asymTrials(iTrial);
                iix = eyeTime > tstart(thisTrial) & eyeTime < tstop(thisTrial);
                
                targetlist = obj.trialTargets{thisTrial};
                
                subplot(sx, sy, iTrial)
                plot(eyePos(iix,1), eyePos(iix,2), 'k'); hold on
                plot(targetlist(:,1), targetlist(:,2), 'sk', 'MarkerSize', 10)
                xlim([-1 1]*12);
                ylim([-1 1]*12);
%                 title(ax3, sprintf('Trial %d', iTrial))

            end
           
           %%
        end
        
       
        
        function update_cmat(obj)
            
            if ~isempty(obj.cmatHat)
                disp('Updating cmat')
                obj.cmat = obj.cmatHat;
            end
        end
        
        function eyePos = get_eyepos(obj, cmat, ix)
            if nargin < 2
                cmat = obj.cmat;
            end
            
            th = cmat(3);
            R = [cosd(th) -sind(th); sind(th) cosd(th)];
            S = [cmat(1) 0; 0 cmat(2)];
            A = (R*S)';
            Ainv = pinv(A);

            eyePos = (obj.Exp.vpx.(obj.raw_field)(:,2:3) - obj.cmat(4:5))*Ainv;
            
            if nargin >2
                eyePos = eyePos(ix,:);
            end

            obj.xy = (obj.Exp.vpx.(obj.raw_field)(obj.validix,2:3) - obj.cmat(4:5))*Ainv;
            
        end
        
        function varargout = refine_calibration(obj)
            
            validTrials = io.getValidTrials(obj.Exp, 'FaceCal');
            
            tstart = obj.Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, obj.Exp.D(validTrials)));
            tstop = obj.Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, obj.Exp.D(validTrials)));
            
            % runs a refinement on the calibration based on 
            eyeTime = obj.Exp.vpx2ephys(obj.Exp.vpx.(obj.raw_field)(:,1));
            eyePos = obj.get_eyepos();
%             th = obj.cmat(3);
%             R = [cosd(th) -sind(th); sind(th) cosd(th)];
%             S = [obj.cmat(1) 0; 0 obj.cmat(2)];
%             A = (R*S)';
%             Ainv = pinv(A);
            
            

%             eyePos = (obj.Exp.vpx.(obj.raw_field)(:,2:3) - obj.cmat(4:5))*Ainv;
            
%             eyePos = obj.Exp.vpx.(obj.raw_field)(:,2:3);
            
            dxdy = diff(eyePos); % velocity
            obj.spd = hypot(dxdy(:,1), dxdy(:,2)); % speed
            
            fixatedLoss = [];
            
            % check if flipped calibrations are better
            lossFlipX = [];
            lossFlipY = [];
            lossFlipXY = [];
            
            fixatedTarget = [];
            fixatedTime = [];
            fixatedXY = [];
            
            fixations = obj.spd < .025; % identify fixations as low-speed moments
            
            for iTrial = 1:numel(tstart)
                iix = eyeTime > tstart(iTrial) & eyeTime < tstop(iTrial);
                
                targetlist = obj.trialTargets{iTrial};
                
                [lp, obj.id] = calibration_loss([1 1 0 0 0], eyePos(iix,:), targetlist);
                % check if flipped calibrations are better
                lpFlipX = calibration_loss([1 1 0 0 0], eyePos(iix,:).*[-1 1], targetlist);
                lpFlipY = calibration_loss([1 1 0 0 0], eyePos(iix,:).*[1 -1], targetlist);
                lpFlipXY = calibration_loss([1 1 0 0 0], eyePos(iix,:).*[-1 -1], targetlist);
                
                
                tax = eyeTime(iix);
                exy = eyePos(iix,:);
                
                fixated = lp < 1 & fixations(iix);
                fixated = imboxfilt(double(fixated), 3) > 0;
                
                fixationList = bwlabel(fixated);
                nFix = max(fixationList);
                
                figure(2); clf
                ax = subplot(2,2,1);
                
                plot(tax, exy(:,1), 'k'); hold on
                
                for iFix = 1:nFix
                    fixix = fixationList == iFix;
                    
                    fixatedTargs = targetlist(obj.id(fixix),:);
                    [~, targId] = min(hypot(fixatedTargs(:,1) - obj.targets(:,1)', fixatedTargs(:,2) - obj.targets(:,2)'),[],2);
                    
                    fixatedLoss = [fixatedLoss; mean(lp(fixix))];
                    
                    % store flipped loss
                    lossFlipX = [lossFlipX; mean(lpFlipX(fixix))];
                    lossFlipY = [lossFlipY; mean(lpFlipY(fixix))];
                    lossFlipXY = [lossFlipXY; mean(lpFlipXY(fixix))];
                    
                    fixatedTarget = [fixatedTarget; mean(targId)];
                    fixatedTime = [fixatedTime; mean(tax(fixix))];
                    fixatedXY = [fixatedXY; mean(exy(fixix,:))];
                    
                    plot(tax(fixix), exy(fixix,1), '.')
                    
                end
                
                ax.YLim = [-10 10];
                
                fixatedTargs = targetlist(obj.id(fixated),:);
                [~, targId] = min(hypot(fixatedTargs(:,1) - obj.targets(:,1)', fixatedTargs(:,2) - obj.targets(:,2)'),[],2);
                
                ax2 = subplot(2,2,3);
                plot(tax, exy(:,2), 'k'); hold on
                plot(tax(fixated), exy(fixated,2), '.r')
                ylabel('horizontal (d.v.a.)')
                ax2.YLim = [-10 10];
                
                ax3 = subplot(2,2,[2 4]);
                plot(exy(:,1), exy(:,2), 'k'); hold on
                fixinds = find(fixated);
                for t = targId(:)'
                    iix = fixinds(t == targId);
                    plot(exy(iix,1), exy(iix,2), '.', 'Color', obj.cmap(t,:))
                end
                ylabel('vertical (d.v.a.)')
                
                plot(targetlist(:,1), targetlist(:,2), 'sk', 'MarkerSize', 20)
                ax3.XLim = [-1 1]*12;
                ax3.YLim = [-1 1]*12;
                title(ax3, sprintf('Trial %d', iTrial))
                
                drawnow
            end
            
            %% check for flips
            L0 = sum(fixatedLoss);
            
            % store flipped loss
            lX = sum(lossFlipX);
            lY = sum(lossFlipY);
            lXY = sum(lossFlipXY);
            
            flips = {[1 1], [-1 1], [1 -1], [-1 -1]};
            
            [~, bestflip] = min([L0 lX lY lXY]);
            
            assert(bestflip==1, 'Calibration flipping is necessary and needs to be implemented')
            
            eyePos = eyePos.*flips{bestflip};
            fixatedXY = fixatedXY.*flips{bestflip};
            
            %% redo calibration using 2nd-order polynomial
            
            targXY = obj.targets(round(fixatedTarget),:);
            features = [fixatedXY fixatedXY.^2];
            
            nFix = numel(fixatedTarget);
            
            nboots = 100;
            ntrain = ceil(nFix/2);
            ntest = nFix - ntrain;
            serror = zeros(ntest, nboots);
            wtsXs = zeros(5, nboots);
            wtsYs = zeros(5, nboots);
            
            for iboot = 1:nboots
                trainindex = randsample(nFix, ntrain, false);
                testindex = setdiff(1:nFix,trainindex);
                
                wtsX = regress(targXY(trainindex,1), [ones(ntrain,1) features(trainindex,:)]);
                wtsY = regress(targXY(trainindex,2), [ones(ntrain,1) features(trainindex,:)]);
                %     wtsX = robustfit(features(trainindex,:), targXY(trainindex,1));%, 'talwar', 1);
                %     wtsY = robustfit(features(trainindex,:), targXY(trainindex,2));%, 'talwar', 1);
                
                exHat = wtsX(1) + features*wtsX(2:end);
                eyHat = wtsY(1) + features*wtsY(2:end);
                
                wtsXs(:,iboot) = wtsX;
                wtsYs(:,iboot) = wtsY;
                
                serror(:,iboot) = hypot( exHat(testindex) - targXY(testindex,1), eyHat(testindex) - targXY(testindex,2));
            end
            
            merror = serror ./ median(serror);
            [~, obj.id] = min(sum(merror > 2));
            
            wtsX = wtsXs(:,obj.id);
            wtsY = wtsYs(:,obj.id);
            
            
            figure(1); clf
            plot(exHat, eyHat,'.'); hold on
            plot(fixatedXY(:,1), fixatedXY(:,2), '.')
            
            [l2, obj.id] = calibration_loss([1 1 0 0 0], [exHat eyHat], unique(targXY, 'rows'));
            l1 = calibration_loss([1 1 0 0 0], fixatedXY, unique(targXY, 'rows'));
            
            [xx,yy] = meshgrid(unique(obj.targets(:)));
            plot(xx, yy, 'k')
            plot(yy, xx, 'k')
            
            %%
            clf
            nFix = size(targXY,1);
            plot(targXY(:,1)+.1*randn(nFix,1), exHat(:,1), '.'); hold on
            plot(xlim, xlim, 'k')
            
            % evaluate
            %
            % iTarg = 13;
            % iix = fixatedTarget == iTarg;
            %
            % [C, ~, ~] = histcounts2(fixatedXY(iix,1)-targets(iTarg,1), fixatedXY(iix,2)-targets(iTarg,2), -2:.1:2, -2:.1:2);
            % [C2, xax, yax] = histcounts2(exHat(iix)-targets(iTarg,1), eyHat(iix)-targets(iTarg,2), -2:.1:2, -2:.1:2);
            % C2 = imgaussfilt(C2,.75);
            %
            % [x,y] = radialcenter(C2');
            % figure(1); clf
            % imagesc(C2'); hold on
            % plot(x,y, 'or')
            %
            % %%
            % figure(1); clf
            % subplot(1,2,1)
            % imagesc(xax, yax, C')
            % hold on
            % plot(xlim, [0 0], 'r')
            % plot([0 0], ylim, 'r')
            %
            % subplot(1,2,2)
            % imagesc(xax, yax, C2')
            % hold on
            % plot(xlim, [0 0], 'r')
            % plot([0 0], ylim, 'r')
            
            
            %%
            eyePosX = wtsX(1) + [eyePos eyePos.^2]*wtsX(2:end);
            eyePosY = wtsY(1) + [eyePos eyePos.^2]*wtsY(2:end);
            
            eyePos2 = [eyePosX eyePosY];
            
            
            % --- find best calibration matrix to match the new calibration
            
            dist = hypot(eyePosX, eyePosY);
            ix = dist < 7;
            
            errfun = @(cmat) sum(sum( (obj.get_eyepos(cmat, ix) - eyePos2(ix,:)).^2 ));
            opts = optimset('Display', 'none');
            obj.cmatHat = fminsearch(errfun, obj.cmat, opts);
            fprintf('Done\n')
            
            if nargout > 0
                varargout{1} = eyePos2;
            end
            
            
        end
        
        function flipX(obj)
           obj.cmat(1) = -obj.cmat(1);
           obj.get_eyepos();
           obj.plot_calibration();
           obj.show_asymetric_trials();
        end
        
        function flipY(obj)
           obj.cmat(2) = -obj.cmat(2);
           obj.get_eyepos();
           obj.plot_calibration();
           obj.show_asymetric_trials();
        end
        
        
        function selectFileDlg(obj)
            
            [filename, pathname] = uigetfile('*.*', 'Pick a data file.');
            
            if filename==0 % 0 when cancel
                fprintf(1, 'File load cancelled by user')
                return
            end
            
            obj.H.filename = fullfile(pathname, filename);
            fprintf('Loading [%s] \n', filename)
            obj.Exp = load(obj.H.filename);
            
            obj.H.Xfilename = strrep(obj.H.filename, '.mat', 'X.csv');
            obj.H.Yfilename = strrep(obj.H.filename, '.mat', 'Y.csv');
            obj.H.Lfilename = strrep(obj.H.filename, '.mat', 'L.csv');
            
            % convert eye position to degrees
            nTrials = numel(obj.Exp.D);
            validTrials = 1:nTrials;
            
            % gain and offsets from online calibration
            cx = cellfun(@(x) x.c(1), obj.Exp.D(validTrials));
            cy = cellfun(@(x) x.c(2), obj.Exp.D(validTrials));
            dx = cellfun(@(x) x.dx, obj.Exp.D(validTrials));
            dy = cellfun(@(x) x.dy, obj.Exp.D(validTrials));
            
            % use the most common value across trials (we should've only calibrated
            % once in these sessions)
            cx = mode(cx);
            cy = mode(cy);
            dx = mode(dx);
            dy = mode(dy);
            
            % x and y position
            vxx = obj.Exp.vpx.(obj.raw_field)(:,2);
            vyy = obj.Exp.vpx.(obj.raw_field)(:,3);
            
            % convert to d.v.a.
            vxx = (vxx - cx)/(dx * obj.Exp.S.pixPerDeg);
            vyy = 1 - vyy;
            vyy = (vyy - cy)/(dy * obj.Exp.S.pixPerDeg);

            obj.xy = [vxx vyy];
            
            obj.xlim = [0 10];
            obj.ylim = [-10 10];
            
            % initialize plots
            obj.plotid = 1;
            
            initialize(obj)
        end

        
       
        function keyboardFcn(obj, ~, k)
            disp(k.Key)
            switch k.Key
                case 'rightarrow'
                    obj.cmat(4) = obj.cmat(4) - .001;
                    th = obj.cmat(3);
                    R = [cosd(th) -sind(th); sind(th) cosd(th)];
                    S = [obj.cmat(1) 0; 0 obj.cmat(2)];
                    A = (R*S)';
                    Ainv = pinv(A);
                    
                    obj.xy = (obj.Exp.vpx.(obj.raw_field)(obj.validix,2:3) - obj.cmat(4:5))*Ainv;
                    
                    plot_calibration(obj);
                    
                case 'leftarrow'
                    obj.cmat(4) = obj.cmat(4) + .001;
                    th = obj.cmat(3);
                    R = [cosd(th) -sind(th); sind(th) cosd(th)];
                    S = [obj.cmat(1) 0; 0 obj.cmat(2)];
                    A = (R*S)';
                    Ainv = pinv(A);
                    
                    obj.xy = (obj.Exp.vpx.(obj.raw_field)(obj.validix,2:3) - obj.cmat(4:5))*Ainv;
                    
                    plot_calibration(obj);
                case 'uparrow'
                    obj.cmat(5) = obj.cmat(5) + .001;
                    th = obj.cmat(3);
                    R = [cosd(th) -sind(th); sind(th) cosd(th)];
                    S = [obj.cmat(1) 0; 0 obj.cmat(2)];
                    A = (R*S)';
                    Ainv = pinv(A);
                    
                    obj.xy = (obj.Exp.vpx.(obj.raw_field)(obj.validix,2:3) - obj.cmat(4:5))*Ainv;
                    
                    plot_calibration(obj);
                    
                case 'downarrow'
                    obj.cmat(5) = obj.cmat(5) - .001;
                    th = obj.cmat(3);
                    R = [cosd(th) -sind(th); sind(th) cosd(th)];
                    S = [obj.cmat(1) 0; 0 obj.cmat(2)];
                    A = (R*S)';
                    Ainv = pinv(A);
                    
                    obj.xy = (obj.Exp.vpx.(obj.raw_field)(obj.validix,2:3) - obj.cmat(4:5))*Ainv;
                    
                    plot_calibration(obj);
                    
                case 'period'
                    obj.cmat(3) = obj.cmat(3) - 1;
                    th = obj.cmat(3);
                    R = [cosd(th) -sind(th); sind(th) cosd(th)];
                    S = [obj.cmat(1) 0; 0 obj.cmat(2)];
                    A = (R*S)';
                    Ainv = pinv(A);
                    
                    obj.xy = (obj.Exp.vpx.(obj.raw_field)(obj.validix,2:3) - obj.cmat(4:5))*Ainv;
                    
                    plot_calibration(obj);
                    
                case 'comma'
                    obj.cmat(3) = obj.cmat(3) + 1;
                    th = obj.cmat(3);
                    R = [cosd(th) -sind(th); sind(th) cosd(th)];
                    S = [obj.cmat(1) 0; 0 obj.cmat(2)];
                    A = (R*S)';
                    Ainv = pinv(A);
                    
                    obj.xy = (obj.Exp.vpx.(obj.raw_field)(obj.validix,2:3) - obj.cmat(4:5))*Ainv;
                    
                    plot_calibration(obj);
                    
                case 'quote'
                    obj.cmat(1) = obj.cmat(1) * .95;
                    th = obj.cmat(3);
                    R = [cosd(th) -sind(th); sind(th) cosd(th)];
                    S = [obj.cmat(1) 0; 0 obj.cmat(2)];
                    A = (R*S)';
                    Ainv = pinv(A);
                    
                    obj.xy = (obj.Exp.vpx.(obj.raw_field)(obj.validix,2:3) - obj.cmat(4:5))*Ainv;
                    
                    plot_calibration(obj);
                    
                case 'l'
                    obj.cmat(1) = obj.cmat(1) * 1.05;
                    th = obj.cmat(3);
                    R = [cosd(th) -sind(th); sind(th) cosd(th)];
                    S = [obj.cmat(1) 0; 0 obj.cmat(2)];
                    A = (R*S)';
                    Ainv = pinv(A);
                    
                    obj.xy = (obj.Exp.vpx.(obj.raw_field)(obj.validix,2:3) - obj.cmat(4:5))*Ainv;
                    
                    plot_calibration(obj);
                    
                case 'semicolon'
                    obj.cmat(2) = obj.cmat(2) * 1.05;
                    th = obj.cmat(3);
                    R = [cosd(th) -sind(th); sind(th) cosd(th)];
                    S = [obj.cmat(1) 0; 0 obj.cmat(2)];
                    A = (R*S)';
                    Ainv = pinv(A);
                    
                    obj.xy = (obj.Exp.vpx.(obj.raw_field)(obj.validix,2:3) - obj.cmat(4:5))*Ainv;
                    
                    plot_calibration(obj);
                    
                case 'p'
                    obj.cmat(2) = obj.cmat(2) * .95;
                    th = obj.cmat(3);
                    R = [cosd(th) -sind(th); sind(th) cosd(th)];
                    S = [obj.cmat(1) 0; 0 obj.cmat(2)];
                    A = (R*S)';
                    Ainv = pinv(A);
                    
                    obj.xy = (obj.Exp.vpx.(obj.raw_field)(obj.validix,2:3) - obj.cmat(4:5))*Ainv;
                    
                    plot_calibration(obj);
            end

        end
            
        
      
        
        function cleanup(obj)
            fclose('all');
        end
        
    end
    
    % --- Callback methods
    methods (Access = public)
        
        
        function labelLost(obj)
            
            if ~isempty(obj.selected)
                obj.Labels(obj.selected) = 4;
            end
                
            % clear selected
            obj.selected = [];
            
            % update plots
            plot_sac_labels(obj)
            
        end
        
        
        
        
        function saveFile(obj)
            
            if isempty(obj.datasetFile)
                [fname, fpath] = uigetfile('*.xls', 'Select the datasets.xls file');
                if isempty(fname)
                    warning('canceled by user')
                    return
                end
                obj.datasetFile = fullfile(fpath, fname);
            end
            
            data = readtable(obj.datasetFile);
            field_list = fields(data);
            if ~any(contains(field_list, 'CalibMat'))
                nSessions = numel(data.Date);
                for i = 1:5
                    data.(sprintf('CalibMat_%d', i)) = nan(nSessions,1);
                end
            end
            
            Tag = strrep(obj.Exp.FileTag, '.mat', '');
            ExNum = find(contains(data.Tag, Tag));
            if isempty(ExNum)
                warning("cannot find experiment. Are you sure you picked the right datasets file?")
            end
            
            if isnan(data.CalibMat_1(ExNum))
                for i = 1:5
                    tmp = data.(sprintf('CalibMat_%d', i));
                    tmp(ExNum) = obj.cmat(i);
                    data.(sprintf('CalibMat_%d', i)) = tmp;
                end
            end
            
            fig = uifigure;
            selection = uiconfirm(fig,'Do you want to save over the existing datasets.xls file values?','Confirm Save',...
                        'Icon','warning');
            close(fig)
            figure(obj.H.fig)
            if strcmp(selection, 'OK')
                fprintf(1, 'Saving data\n')
                writetable(data, obj.datasetFile);
                fprintf(1, 'Done\n')
            else
                fprintf(1, 'Canceled\n')
                
            end
            

            
        end
        
        
    end
    
    methods(Static)
        
       
        
    end
    
end


