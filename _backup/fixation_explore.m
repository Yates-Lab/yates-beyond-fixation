%%
stimulusSet = 'BackImage';
validTrials = io.getValidTrials(Exp, stimulusSet);

tstart = Exp.ptb2Ephys(cellfun(@(x) x.STARTCLOCKTIME, Exp.D(validTrials)));
tstop = Exp.ptb2Ephys(cellfun(@(x) x.ENDCLOCKTIME, Exp.D(validTrials)));

% fixation times
fixon = Exp.vpx2ephys(Exp.slist(1:end-1,2));
sacon = Exp.vpx2ephys(Exp.slist(2:end,1));


eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1));
eyeX = Exp.vpx.smo(:,2);
eyeY = Exp.vpx.smo(:,3);
eyesmoothing = 19;
    
eyeX = sgolayfilt(eyeX, 1, eyesmoothing);

eyeY = sgolayfilt(eyeY, 1, eyesmoothing);

iTrial = 1;
%%

iTrial = iTrial + 1;
        
fprintf('%d/%d\t', iTrial, numel(validTrials))
        
thisTrial = validTrials(iTrial);
        
% load image
try
    Im = imread(fullfile(fileparts(which('marmoV5')), Exp.D{thisTrial}.PR.imagefile));
catch
    try
        Im = imread(fullfile(fileparts(which('marmoV5')), strrep(Exp.D{thisTrial}.PR.imageFile, '\', filesep)));
    catch
        error('regenerateStimulus: failed to load image\n')
    end
end
        
% zero mean
Im = mean(Im,3)-127;
Im = imresize(Im, fliplr(Exp.S.screenRect(3:4)));
        
figure(1); clf
imagesc(Im); hold on
colormap gray

eyeix = eyeTime > tstart(iTrial) & eyeTime < tstop(iTrial);

figure(2); clf
ax = axes();
plot(eyeTime(eyeix), eyeX(eyeix)); hold on
plot(eyeTime(eyeix), eyeY(eyeix));
        
% fprintf('%d fixations...\n', nft)
% loop over fixations

fixix = find(fixon > tstart(iTrial) & fixon < tstop(iTrial));
nfix = numel(fixix);

for ifix = 1:nfix
    x = [fixon(fixix(ifix))*[1 1] sacon(fixix(ifix))*[1 1]];
    y = [ylim fliplr(ylim)];
    fill(x,y,'r', 'FaceAlpha', .1)
%     fill(*[1 1], ylim, 'r')
end


% ax2 = axes('Position', ax.Position);

ifix = 0;

spkix = Exp.osp.st > tstart(iTrial) & Exp.osp.st < tstop(iTrial);
st = Exp.osp.st(spkix);
clu = Exp.osp.clu(spkix);
plot.raster(st, clu+10, 2, 'k', 'Linewidth', 2)
axis off



%%
win = [0.05 0];
ppd = Exp.S.pixPerDeg;
ctr = Exp.S.centerPix;
rect = [-1 -1 1 1]*ceil(ppd*1); % window centered on RF
dims = [rect(4)-rect(2) rect(3)-rect(1)];
hwin = hanning(dims(1))*hanning(dims(2))';

ifix = ifix + 1;
if ifix > nfix
    ifix = 0;
end
            
thisfix = fixix(ifix);

ii = eyeTime > fixon(thisfix)+win(1) & eyeTime < sacon(thisfix)+win(2);

et = eyeTime(ii);
fixX = eyeX(ii)*ppd + ctr(1);
fixY = -eyeY(ii)*ppd + ctr(2);

% center on eye position
i = ceil(numel(et)/2);
tmprect = rect + [fixX(i) fixY(i) fixX(i) fixY(i)];

imrect = [tmprect(1:2) (tmprect(3)-tmprect(1))-1 (tmprect(4)-tmprect(2))-1];
I = imcrop(Im, imrect); % requires the imaging processing toolbox

figure(1); clf

% plot(et, eyeX(ii))


nbins = 500;
binedges = linspace(fixon(thisfix)+win(1), sacon(thisfix)+win(2), nbins);
cmap = parula(nbins);

[~, ~, binid] = histcounts(et, binedges);

ax = subplot(2,2,1:2);
xx = (fixX-ctr(1))/ppd*60;
x0 = mean(xx);
plot(et-fixon(thisfix), xx-x0, 'k'); hold on
for i = 1:nbins
    plot(et(binid==i)-fixon(thisfix), (fixX(binid==i)-ctr(1))/ppd*60-x0, '.', 'MarkerSize', 10, 'Color', cmap(i,:));
end
xd = xlim;

figure(2); clf
imagesc(Im); colormap gray
% imagesc(rect(1):rect(3), rect(2):rect(4), I)
hold on
for i = 1:nbins
    plot(fixX(binid==i), fixY(binid==i), '.', 'MarkerSize', 10, 'Color', cmap(i,:));
end


% plot(fixX, fixY, 'r', 'Linewidth', 2)
xlim(tmprect([1 3]))
ylim(tmprect([2 4]))


iix = st > fixon(thisfix)+win(1) & st < sacon(thisfix)+win(2);
[cnt, ~, binid] = histcounts(st(iix), binedges);


figure(1)
subplot(2,1,1)
plot(binedges(1:end-1)-fixon(thisfix), imgaussfilt(cnt, 2), 'k')

subplot(2,1,2)

st_ = st(iix);
clu_ = clu(iix);

for i = 1:nbins
    plot.raster(st_(binid==i)-fixon(thisfix), clu_(binid==i), 5, 'Linewidth', 2, 'Color', cmap(i,:)); hold on
end
xlim(xd)



%%
Iwin = (I - mean(I(:))).*hwin;

imagesc(Iwin, [-1 1]*max(abs(Iwin(:))))
axis off

%%
fIm = fftshift(fft2(Iwin, nxfft, nyfft));
            
            
            



%%
[valid, epoch] = getTimeIdx(fixon, tstart, tstop);
fixon = fixon(valid);
sacon = sacon(valid);
fixTrial = epoch(valid);

fixdur = sacon - fixon;

% --- eye position
eyeTime = Exp.vpx2ephys(Exp.vpx.smo(:,1)); % time
remove = find(diff(eyeTime)==0); % bad samples

%%



%%
% filter eye position with 1st order savitzy-golay filter
eyeX = sgolayfilt(Exp.vpx.smo(:,2), 1, 31);
eyeX(isnan(eyeX)) = 0;
eyeY = sgolayfilt(Exp.vpx.smo(:,3), 1, 31);
eyeY(isnan(eyeY)) = 0;

% remove bad samples
eyeTime(remove) = [];
eyeX(remove) = [];
eyeY(remove) = [];

binSize = median(diff(eyeTime));

[~, ~, id1] = histcounts(fixon, eyeTime);
[~, ~, id2] = histcounts(sacon, eyeTime);


nFix = numel(fixon);
drifts = cell(nFix,1);
for ifix = 1:nFix
    
    
    
    fixix = id1(ifix):id2(ifix);
    fixix = fixix(30:end);
    
    if numel(fixix) < 200
        continue
    end
    
    ctrx = eyeX(fixix(1));
    ctry = eyeY(fixix(1));

    
    fixX = (eyeX(fixix)-ctrx)*60;
    fixY = (eyeY(fixix)-ctry)*60;
    spd = hypot(diff(fixX), diff(fixY));
    nxsac = find(spd > 1, 1);
    if isempty(nxsac)
        nxsac = numel(spd);
    end
    nxsac = nxsac - 15;
    
    fixX = fixX(1:nxsac);
    fixY = fixY(1:nxsac);
    
    drift = hypot(fixX, fixY);
    drifts{ifix} = drift;
end