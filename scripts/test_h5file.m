
fname = '~/Dropbox/Datasets/Mitchell/stim_movies/logan_20200304_-31_-2_49_78_1_1_1_41_0_0.hdf5';


%% test that it worked
id = 1;
stim = 'Gabor';
tset = 'Train';

sz = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'size');

iFrame = 1;

%% show sample frame

iFrame = iFrame + 1;
I = h5read(fname, ['/' stim '/' tset '/Stim'], [iFrame, 1,1], [1 sz(1:2)']);
I = squeeze(I);
% I = h5read(fname{1}, ['/' stim '/' set '/Stim'], [1,1,iFrame], [sz(1:2)' 1]);
figure(id); clf
imagesc(I)
colorbar
colormap gray
drawnow



%% get STAs to check that you have the right rect
spike_sorting = 'kilo';
Stim = h5read(fname, ['/' stim '/' tset '/Stim']);
% Robs = h5read(fname, ['/' stim '/' set '/Robs']);
ftoe = h5read(fname, ['/' stim '/' tset '/frameTimesOe']);

frate = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'frate');
st = h5read(fname, ['/Neurons/' spike_sorting '/times']);
clu = h5read(fname, ['/Neurons/' spike_sorting '/cluster']);
cids = h5read(fname, ['/Neurons/' spike_sorting '/cids']);
sp = struct();
sp.st = st;
sp.clu = clu;
sp.cids = cids;
Robs = binNeuronSpikeTimesFast(sp, ftoe-8e-3, 1/frate);

% Robs = 
eyeAtFrame = h5read(fname, ['/' stim '/' tset '/eyeAtFrame']);
labels = h5read(fname, ['/' stim '/' tset '/labels']);
NX = size(Stim,2);
NY = size(Stim,3);
NC = size(Robs,2);

Stim = reshape(Stim, size(Stim, 1), NX*NY);
Stim = zscore(single(Stim));

%% Pick a lag and compute the STA quickly for all cells 
figname = 'results2.pdf';
lag = 8;
ecc = hypot(eyeAtFrame(:,2)-Exp.S.centerPix(1), eyeAtFrame(:,3)-Exp.S.centerPix(2))/Exp.S.pixPerDeg;
ix = ecc < 5.2 & labels == 1 & (1:numel(ecc))'> lag;
Rdelta = Robs - mean(Robs);
Rdelta = Rdelta(ix,:);
sta = (Stim(find(ix)-lag,:).^2)'*Rdelta;
% sta = (Stim(find(ix)-lag,:))'*Rdelta;
[~, ind] = sort(std(sta));

sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(10); clf
for cc = 1:NC
    subplot(sx, sy, cc, 'align')
    imagesc(reshape(sta(:,ind(cc)), [NX NY]))
    axis off
end

set(gcf, 'PaperSize', [7 7], 'PaperPosition', [0 0 7 7])
exportgraphics(gcf, figname); 

figure(11); clf
plot(std(sta(:,ind)), '-o')
cids = find(std(sta) > 500); 

set(gcf, 'PaperSize', [4 3], 'PaperPosition', [0 0 4 3])
exportgraphics(gcf, figname, 'Append', true);


%%

