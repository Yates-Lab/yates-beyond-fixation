%%

fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'FixFlashGabor', 'Gabor', 'BackImage'}, 'GazeContingent', true, 'includeProbe', true);

%% no gaze correction, no fixation point
fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'FixFlashGabor'}, 'GazeContingent', false, 'includeProbe', false);

%% gaze correction, no fixation point
fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'FixFlashGabor'}, 'GazeContingent', true, 'includeProbe', false);
%%
% fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'FixFlashGabor'}, 'GazeContingent', false, 'includeProbe', true);

%%
ftag = strrep(Exp.FileTag, '.mat', '');
flist = dir(['~/Dropbox/Datasets/Mitchell/stim_movies/' ftag '*']);

for i = 1:numel(flist)
    fname = fullfile(flist(i).folder, flist(i).name);
    io.h5addSpikesorting(Exp, fname, 'kilo');
end


%%
%% test that it worked
id = 1;
stim = 'FixFlashGabor';
tset = 'Train';

sz = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'size');

iFrame = 1;

%% show sample frame

iFrame = iFrame + 1;
I = h5read(fname, ['/' stim '/' tset '/Stim'], [iFrame, 1,1], [1 sz(1:2)']);
I = squeeze(I);
% I = h5read(fname{1}, ['/' stim '/' set '/Stim'], [1,1,iFrame], [sz(1:2)' 1]);
figure(id); clf
subplot(1,2,1)
imagesc(I)
subplot(1,2,2)
imagesc(I(1:2:end,1:2:end));% axis xy
colorbar
colormap gray
drawnow



%% get STAs to check that you have the right rect
spike_sorting = 'kilowf';
Stim = h5read(fname, ['/' stim '/' tset '/Stim']);
% Robs = h5read(fname, ['/' stim '/' set '/Robs']);
ftoe = h5read(fname, ['/' stim '/' tset '/frameTimesOe']);

frate = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'frate');
st = h5read(fname, ['/Neurons/' spike_sorting '/times']);
clu = h5read(fname, ['/Neurons/' spike_sorting '/cluster']);
cids = h5read(fname, ['/Neurons/' spike_sorting '/cids']);
sp.st = st;
sp.clu = clu;
sp.cids = cids;
Robs = binNeuronSpikeTimesFast(sp, ftoe-8e-3, 1/frate);

% Robs = 
eyeAtFrame = h5read(fname, ['/' stim '/' tset '/eyeAtFrame']);
labels = h5read(fname, ['/' stim '/' tset '/labels']);
NX = size(Stim,2);
NY = size(Stim,3);
%%

Stim = reshape(Stim, size(Stim, 1), NX*NY);

% reshape(Stim, 
Stim = zscore(single(Stim));

%% forward correlation
NC = size(Robs,2);
nlags = 20;
Rdelta = Robs - mean(Robs);
nstim = size(Stim,2);
ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc < 5.2 & labels == 1;

stas = zeros(nlags, nstim, NC);
for idim = 1:nstim
    fprintf('%d/%d\n', idim, nstim)
    Xstim = conv2(Stim(:,idim), eye(nlags), 'full');
    Xstim = Xstim(1:end-nlags+1,:);
    stas(:, idim, :) = Xstim(ix,:)'*Rdelta(ix,:);
end

% %%
% % only take central eye positions
% ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
% ix = ecc > 5.1 | labels ~= 1;
% 
% NC = size(Robs,2);
% nlags = 10;
% NT = size(Stim,1);
% sx = ceil(sqrt(NC));
% sy = round(sqrt(NC));
% 
% figure(1); clf
% stas = zeros(nlags, size(Stim,2), NC);
% 
% for cc = 1:NC
%     if sum(Robs(:,cc))==0
%         continue
%     end
%     rtmp = Robs(:,cc);
%     rtmp(ix) = 0; % remove spikes we don't want to analyze
%     rtmp = rtmp - mean(rtmp);
%     sta = simpleRevcorr(Stim, rtmp, nlags);
%     subplot(sx, sy, cc, 'align')
%     plot(sta)
%     stas(:,:,cc) = sta;
%     drawnow
% end
% 
% %%
% % clearvars -except Exp Stim Robs
    %%
cc = 0;

%% plot one by one
figure(2); clf
cc = cc + 1;
if cc > NC
    cc = 1;
end

sta = stas(:,:,cc);
% NY = size(Stim,2)/NX;
% sta = (sta - min(sta(:))) ./ (max(sta(:)) - min(sta(:)));
sta = (sta - mean(sta(:))) ./ std(sta(:));
% x = xax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
% y = yax(1:opts.s_downsample:end)/Exp.S.pixPerDeg*60;
% xax = (1:NX)/Exp.S.pixPerDeg*60;
% yax = (1:NY)/Exp.S.pixPerDeg*60;
for ilag = 1:nlags
   subplot(2,ceil(nlags/2), ilag, 'align')
   imagesc(reshape(sta(ilag,:), [NX NY])', [-1 1]*4)
end

% colormap(plot.viridis)
colormap(gray)
title(cc)