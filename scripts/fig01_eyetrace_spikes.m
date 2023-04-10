
%% get sessions
datadir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'preprocessed');
flist = dir(fullfile(datadir, '*.mat'));

%% Load Session
sessnum = 1;
Exp = load(fullfile(datadir, flist(sessnum).name));

%% Plot example trial

validTrials = io.getValidTrials(Exp, 'BackImage');

i = 1; % trial number

iTrial = validTrials(i);

tstart = Exp.ptb2Ephys(Exp.D{iTrial}.STARTCLOCKTIME);
tstop = Exp.ptb2Ephys(Exp.D{iTrial}.ENDCLOCKTIME);

ttime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
iix = ttime > tstart & ttime < tstop;

eyet = ttime(iix) - tstart;
eyex = sgolayfilt(Exp.vpx.smo(iix,2), 1, 19);
eyey = sgolayfilt(Exp.vpx.smo(iix,3), 1, 19);
labels = Exp.vpx.Labels(iix);

figure(1); clf
plot(eyet, eyex, 'k'); hold on

showY = true;

if showY
    plot(eyet, eyey, 'Color', .5*[1 1 1])
end

yd = [-1 1]*10;
cmap = lines;

% plot saccades
saccades = bwlabel(labels==2);
sacnum = unique(saccades);
for isac = 1:max(sacnum)
    ii = saccades == isac;
    plot(eyet(ii), eyex(ii), 'r')
    if showY
        plot(eyet(ii), eyey(ii), 'r')
    end
end

% plot blinks
blinks = bwlabel(labels==4);
blinknum = unique(blinks);
for iblink = 1:max(blinknum)
   ii = find(blinks==iblink);
   fill( eyet(ii([1 1 end end])), yd([1 2 2 1]), 'k', 'FaceColor', cmap(1,:), 'FaceAlpha', .5, 'EdgeColor', 'none');
end


plot([0 0]-.1, yd(1) + [0 5], 'k', 'Linewidth', 2)
plot([0 1]-.1, yd(1)*[1 1], 'k', 'Linewidth', 2)



cids = Exp.osp.cids;
NC = numel(cids);
iisp = Exp.osp.st > tstart & Exp.osp.st < tstop & ismember(Exp.osp.clu, cids);
st = Exp.osp.st(iisp);
clu = Exp.osp.clu(iisp);

cidmap = zeros(max(cids), 1);
cidmap(cids) = 1:NC;

ii = clu~=0;
plot.raster(st(ii)-tstart, yd(2)*1.5 + cidmap(clu(ii)), 1, 'k')

% plot fixation
fixations = bwlabel(labels==1);
fixnum = unique(fixations);
for ifix = 1:max(fixations)
    ii = find(fixations == ifix);
    ii = ii(50:end);
    if isempty(ii)
        continue
    end
    fill( eyet(ii([1 1 end end])), yd(2)*1.5 + double(cids([1 NC NC 1])), 'k', 'FaceColor', cmap(5,:), 'FaceAlpha', .25, 'EdgeColor', 'none');
    
%     plot(eyet(ii), eyex(ii), 'Color', cmap(5,:))
%     if showY
%         plot(eyet(ii), eyey(ii), 'Color', cmap(5,:))
%     end
    
end

ylim([-10 50])
axis off

plot.formatFig(gcf, [4.5 1], 'nature')
saveas(gcf, fullfile('Figures/fig01_trace.pdf'))
