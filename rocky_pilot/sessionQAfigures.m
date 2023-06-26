function sessionQAfigures(Exp, S)
% run several quality assessment analyses and save out figures
% sessionQAfigures(Exp)

if nargin < 2
    warning('using latency of 0')
    S.Latency = 8e-3;
end

%% find specific trial lists
trialProtocols = cellfun(@(x) x.PR.name, Exp.D, 'uni', 0);
protocols = unique(trialProtocols);
numProtocols = numel(protocols);
fprintf('Session was run with %d protocols\n', numProtocols);
for i = 1:numProtocols
    
    nTrials = sum(strcmp(trialProtocols, protocols{i}));
    fprintf('%d) %d trials of %s\n', i, nTrials, protocols{i});
end

%% start report
fname = fullfile('Figures', 'sessionQA', strrep(Exp.FileTag, '.mat', '_QA'));
       
fixFlashTrials = find(strcmp(trialProtocols, 'FixFlash'));
goodTrials = cellfun(@(x) x.PR.error, Exp.D(fixFlashTrials)) == 0;
fixFlashTrials(~goodTrials) = [];

if numel(fixFlashTrials) > 5
    
    eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
    bins = {-2:.05:2, -2:.05:2};
    C = hist3 ([inf inf], bins)';
    xyFull = [];
    for iTrial = 1:numel(fixFlashTrials)
        thisTrial = fixFlashTrials(iTrial);
        
        fixS = diff(Exp.D{thisTrial}.eyeData(:,5)==1);
        % fixation onset in ephys time
        fixOn = Exp.ptb2Ephys(Exp.D{thisTrial}.eyeData(find(fixS>0, 1),1));
        fixOff = Exp.ptb2Ephys(Exp.D{thisTrial}.eyeData(find(fixS<0, 1),1));
        
        eyeIx = eyeTime > fixOn & eyeTime < fixOff;
        xy = Exp.vpx.smo(eyeIx,2:3);
        C = C + hist3 (xy, 'Ctrs', bins)';
        
        xyFull = [xyFull; xy]; %#ok<AGROW>
    end
    
    f = figure(1); clf
    h = imagesc(bins{1}, bins{2}, imgaussfilt(C,1));
    % contourf(bins{1}, bins{2}, imgaussfilt(C,.5));
    h.AlphaData = sqrt(h.CData);
    axis xy
    hold on
    grid on
    set(gca, 'GridAlpha', 0.5, ...
        'GridColor', 'k', ...
        'GridLineStyle', '-')
    plot(0,0,'r+', 'MarkerSize', 20, 'Linewidth', 2)
    title('Eye position during FixFlash')
    xlabel('d.v.a')
    ylabel('d.v.a')
    xlim([-2 2])
    ylim([-2 2])
    
    plot(xyFull(:,1), xyFull(:,2), 'k.')
    mx = mean(xyFull(:,1));
    my = mean(xyFull(:,2));
    quiver(0, 0, mx, my, 'r', 'Linewidth', 2)
    text(1, 1, sprintf('x: %02.2f', mx), 'Color', 'r')
    text(1, .8, sprintf('y: %02.2f', my), 'Color', 'r')
    
   

end

colormap(parula)
%%
% Quick spatial RF map
S = struct();
S.Latency = 0;
rect = [-200 -200 200 200];
binSize = .5*Exp.S.pixPerDeg; % we just want to find the rect

[RF, ~] = quick_sta_rfs(Exp, 'ROI', rect, 'binSize', binSize, 'latency', S.Latency);

if isstruct(RF)
    
%     f = figure(1);

    
    % save figure
    plot.fixfigure(figure(1), 8, [10 10], 'OffsetAxes', false)
    saveas(figure(1), fullfile('Figures', 'sessionQA', strrep(Exp.FileTag, '.mat', '_quickRF.pdf')))
    
%     save figure 2
    plot.fixfigure(figure(2), 8, [5 5], 'OffsetAxes', false)
    saveas(figure(2), fullfile('Figures', 'sessionQA', strrep(Exp.FileTag, '.mat', '_rfRect.pdf')))
end

%% Eye traces QA

% add report chapter head
f = figure(1); clf
xs = Exp.vpx.smo(Exp.slist(:,4),2);
ys = Exp.vpx.smo(Exp.slist(:,4),3);
xe = Exp.vpx.smo(Exp.slist(:,5),2);
ye = Exp.vpx.smo(Exp.slist(:,5),3);

dx = xe-xs;
dy = ye-ys;
dd =  hypot(dx, dy);

nSaccades = size(Exp.slist,1);
isi = diff(Exp.slist(:,1));
h = histogram(isi, 'binEdges', 0:.01:1);
xlabel('ISI (sec)')
ylabel('Count')
text(.8, .9*max(h.Values), sprintf('n = %d', nSaccades))
m = prctile(isi(isi < 1), [16 50 84]);
hold on
cmap = lines;
plot(m(1)*[1 1], ylim, '--', 'Linewidth', 2, 'Color', cmap(4,:))
plot(m(2)*[1 1], ylim, '', 'Linewidth', 2, 'Color', cmap(4,:))
plot(m(3)*[1 1], ylim, '--', 'Linewidth', 2, 'Color', cmap(4,:))
%%
plot.fixfigure(figure(1), 8, [5 3])
saveas(figure(1), fullfile('Figures', 'sessionQA', strrep(Exp.FileTag, '.mat', '_SaccadeISI.pdf')))


% Saccade amplitude and velocity
%%

f = figure(1); clf
dt = Exp.vpx.smo(Exp.slist(:,5),1) - Exp.vpx.smo(Exp.slist(:,4),1);
v = Exp.vpx.smo(Exp.slist(:,6),7);
dd(isnan(v)) = [];
v(isnan(v)) = [];

bins = {0:.5:20, 0:10:1.2e3};
C = hist3([dd v], 'Ctrs', bins);
C = imgaussfilt(C,.25)';

p = log(C);
p(isinf(p)) = 10e-6;

fun = @(w, x) (w(1)*x).^w(2);

evalc('w = lsqcurvefit(fun, [1 1], dd, v);'); % suppress output

subplot(1,2,1)
% [~, h] = contourf(bins{1}, bins{2}, p, [.5:4:max(p(:))]);
% h.LineColor = 'none';

h = imagesc(bins{1}, bins{2}, p);
h.AlphaData = abs(sqrt(C));
axis xy
colormap([ones(70,3); winter(50)])
xlabel('Amplitude (d.v.a)')
ylabel('Peak Velocity (d.v.a / sec)')
hold on
plot(bins{1}, fun(w, bins{1}), 'r', 'Linewidth', 2)
xlim([0 20])
ylim([0 1e3])
subplot(1,2,2)
h2 = plot(dd, v, 'ow', 'MarkerFaceColor', 'b', 'MarkerSize', 2); hold on
plot(bins{1}, fun(w, bins{1}), 'r', 'Linewidth', 2)
xlabel('Amplitude (d.v.a)')
ylabel('Peak Velocity (d.v.a / sec)')
xlim([0 20])
ylim([0 1e3])
text(0, 900, sprintf('slope = %02.2f', w(1)))
text(0, 800, sprintf('exponent = %02.2f', w(2)))



plot.fixfigure(figure(1), 8, [5 3])
% saveas(figure(1), fullfile('Figures', 'sessionQA', strrep(Exp.FileTag, '.mat', '_SaccadeMetrics.pdf')))

%% plot example trace with labeling


f = figure(1); clf
L = Exp.vpx.Labels;

x = Exp.vpx.smo(:,2);
% plot(x, 'k'); hold on
cmap = lines(4);

figure(1); clf
[~, id] = max(filter(ones(1e3,1), 1, double(L==1)));
window = -2e3:2e3;
t = Exp.vpx.smo(id + window,1);
x =  x(id + window);
plot(t,x, 'k')
axis tight
ylim([-10 10]); 
hold on
L = L(id + window);
for i = 1:2
    plot(t(L==i), x(L==i), '.', 'Color', cmap(i,:)); hold on
end


f = figure(1); clf

bins = {-20:.25:20, -20:.25:20};
C = hist3(Exp.vpx.smo(:,2:3), 'Ctrs', bins );
imagesc(bins{1}, bins{2}, log(C'))
xlim([-18 18])
ylim([-18 18])
axis xy
xlabel('d.v.a')
ylabel('d.v.a')
title('Position')
hold on
plot([0 0], ylim, 'r');
hold on
plot(xlim, [0 0], 'r');
colormap(parula)


f = figure(1); clf

bins = {-20:.25:20, -20:.25:20};
C = hist3([dx dy], 'Ctrs', bins);

imagesc(bins{1}, bins{2}, (C')); hold on
% plot(dx, dy, '.')
h2 = plot(dx, dy, '.r', 'MarkerFaceColor', 'b', 'MarkerSize', 2); hold on
colorbar
xlim([-5 5])
ylim([-5 5])
xlabel('d.v.a')
ylabel('d.v.a')
title('Saccade Endpoint')
axis xy
grid on



plot.fixfigure(figure(1), 8, [6 3])
saveas(figure(1), fullfile('Figures', 'sessionQA', strrep(Exp.FileTag, '.mat', '_EyePosDistribution.pdf')))

%% tremor?



x = Exp.vpx.smo(:,2);
% xs = sgolayfilt(x, 2, 11);
sd = 9;
xs = imgaussfilt(x, sd);
% xs = imboxfilt(x, 11);
L = Exp.vpx.Labels;

fixations = L==1;
[bw, n] = bwlabel(fixations);
n = min(1e3,n); % take the first thousand
ar = nan(n,1);
for i = 1:n % first thoustand
    ar(i) = sum(bw==i);
end

iFix = find(ar>200,1);

ix = bw==iFix;
f = figure(1); clf
t =Exp.vpx.smo(ix,1);
t = t-t(1);
plot(t*1e3,  (x(ix)-nanmean(x(ix)))*60); hold on
plot(t*1e3, (xs(ix)-nanmean(x(ix)))*60);
xlabel('Time (ms)')
ylabel('arcmin')



%%
f = figure(1); clf
xd = x - xs;
ix = L==1 & ~isnan(xd) & xd.^2 < 0.01; 
xd = xd(ix);
histogram(xd*60, 'Normalization', 'pdf', 'Linestyle', 'none')
xlabel('arcmin')
ylabel('Probability')

%%
figure(2); clf
Fs = 1./median(diff(Exp.vpx.smo(:,1)));
pwelch(xd, [], [], [], Fs)

