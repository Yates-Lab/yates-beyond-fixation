%% add paths

user = 'jakelaptop';
addFreeViewingPaths(user);

% addpath Analysis/202001_K99figs_01
addpath Analysis/manuscript_freeviewingmethods/
figDir = 'Figures/manuscript_freeviewing/fig07';
%% Load analyses from prior steps

sesslist = io.dataFactory;
sesslist = sesslist(1:57); % exclude monash sessions

% Spatial RFs
sfname = fullfile('Data', 'spatialrfsreg.mat');
load(sfname)

% Grating RFs
fittype = 'basis';
gfname = fullfile('Data', sprintf('gratrf_%s.mat', fittype));
load(gfname)

fftrf = cell(numel(sesslist),1);

fftname = fullfile('Data', sprintf('fftrf_%s.mat', fittype));
load(fftname)

fixname = fullfile('Data', 'fixrate.mat');

load(fixname)

wname = fullfile('Data', 'waveforms.mat');
load(wname)

%%

ix = sigs & sigg;
ix = ix & wfamp > 40;

lags = fftrf{1}.rfs_post(1).lags;
iix = lags>-.05 & lags<0; 

mnorm = max(nanmean(mrateHi(:,iix),2), 1);

nrateHi = mrateHi ./ mnorm;
nrateLow = mrateLow ./ mnorm;

figure(1); clf
subplot(1,2,1)
imagesc(nrateHi(ix,:))
subplot(1,2,2)
imagesc(nrateLow(ix,:))

figure(2); clf
plot(nrateHi(ix,:)', 'b'); hold on
plot(nrateLow(ix,:)', 'r');


rdiff = nrateHi(ix,:) - nrateLow(ix,:);

figure(3); clf
plot(lags, rdiff')


% plot(sfPref(ix), mean(nrateHi(ix,100:200),2), 'o')

Bearly = exp(- ((lags*1e3 - 60).^2/2/20^2)); Bearly = Bearly ./ sum(Bearly);
Blate = 1 ./ (1 + exp(- (lags*1e3 - 100) / 20)); Blate = Blate ./ sum(Blate);

plot(Bearly); hold on
plot(Blate)


%% load data
iEx = 45;
Exp = io.dataFactory(45, 'spike_sorting', 'kilowf');

eyePosOrig = Exp.vpx.smo(:,2:3);

eyePos = io.getCorrectedEyePos(Exp, 'plot', true, 'usebilinear', false);
lam = .5; % mixing between original eye pos and corrected eye pos
Exp.vpx.smo(:,2:3) = lam*eyePos + (1-lam)*Exp.vpx.smo(:,2:3);

%%

[rf_post, plotmeta] = fixrate_by_fftrf(Exp, Srf{iEx}, Sgt{iEx}, 'debug', false, 'plot', true, 'usestim', 'post', 'alignto', 'fixon');

%%
iEx = 46;
cc = cc + 1;
if cc > numel(fftrf{iEx}.rfs_post)
    cc = 1;
end
figure(1); clf
sfs = fftrf{iEx}.rfs_post(cc).sftuning.sfs;
sfix = ismember(sfs, [1 10]);
lags = fftrf{iEx}.rfs_post(cc).lags;
sftuning = fftrf{iEx}.rfs_post(cc).sftuning.rhos'*fftrf{iEx}.rfs_post(cc).sftuning.B;
psthhi = fftrf{iEx}.rfs_post(cc).sftuning.sfpsthhi(:,sfix);
psthlow = fftrf{iEx}.rfs_post(cc).sftuning.sfpsthlow(:,sfix);

subplot(1,2,1)
plot(lags, psthhi, 'Linewidth', 2)
title(cc)


subplot(1,2,2)

plot(sfs, sftuning)



%%

ar = []; % sqrt area (computed from gaussian fit)
ecc = []; % eccentricity
maxV = []; % volume of RF blob

sfPref = [];  % spatial frequency preference
sfBw = [];    % spatial frequency bandwidth (FWHM)
oriPref = []; % orientation preference
oriBw  = [];  % orientation bandwidth (FWHM)

sigg = []; % boolean: grating RF is significant
sigs = []; % boolean: spatial RF is significant

r2 = [];   % r-squared from gaussian fit to RF
gtr2 = []; % r-squared of parametric fit to frequecy RF

ctr = []; % counter for tracking cell number
cgs = []; % cluster quality

mshift = []; % how much did the mean shift during fitting (measure of whether we got stuck in local minimum)

% FFT modulations
mrateHi = [];
mrateLow = [];
fftcorr = [];
fftpBY = [];
fftpBH = [];
fftpval = [];

% freq dependent
sfs = fftrf{1}.rfs_post(1).sftuning.sfs;
sfix = ismember(sfs, [1 10]);
lags = fftrf{1}.rfs_post(1).lags;

sftuningE = [];
sftuningL = [];
spsthhi1 = [];
spsthhi10 = [];
spsthlow1 = [];
spsthlow10 = [];

field = 'rfs_post';

% fixrate modulations
nipeakt = []; % time of peak for nat images
nipeakv = []; % value at peak (in relative rate) for nat images
nitrot = []; % trough time
nitrov = []; % trough value
nimrate = []; % mean rate
grpeakt = []; % time of peak for gratings
grpeakv = []; % value at peak (in relative rate) for gratings
grtrot = []; % trough time
grtrov = []; % trough value
grmrate = []; % mean rate

rflag = []; % peak lag of the temporal RF

wf = [];

zthresh = 8;
for ex = 1:numel(Srf)
    
    if isempty(fftrf{ex})
        continue
    end
    srf = Srf{ex};
    if isfield(srf, 'fine')
        srf = srf.fine;
    end
    
    if ~isfield(srf, 'rffit') || ~isfield(Sgt{ex}, 'rffit') || (numel(Sgt{ex}.rffit) ~= numel(srf.rffit))
        continue
    end
    
    NC = numel(srf.rffit);
    for cc = 1:NC
        if ~isfield(srf.rffit(cc), 'mu')
            continue
        end
         
         if isempty(Sgt{ex}.rffit(cc).r2) || isempty(srf.rffit(cc).r2)
             continue
         end
         
%          maxV = [maxV; srf.maxV(cc)];
         
         % Tuning preferences
         oriPref = [oriPref; Sgt{ex}.rffit(cc).oriPref];
         oriBw = [oriBw; Sgt{ex}.rffit(cc).oriBandwidth];
         sfPref = [sfPref; Sgt{ex}.rffit(cc).sfPref];
         sfBw = [sfBw; Sgt{ex}.rffit(cc).sfBandwidth];
         gtr2 = [gtr2; Sgt{ex}.rffit(cc).r2];
             
         % significance
         sigg = [sigg; Sgt{ex}.sig(cc)];
         sigs = [sigs; srf.sig(cc)];
        
         r2 = [r2; srf.rffit(cc).r2]; % store r-squared
         ar = [ar; srf.rffit(cc).ar];
         ecc = [ecc; srf.rffit(cc).ecc];
        
         rflag = [rflag; Sgt{ex}.peaklagt(cc)];
         % unit quality metric
         cgs = [cgs; srf.cgs(cc)];
        
         mshift = [mshift; srf.rffit(cc).mushift]; %#ok<*AGROW>
         
         % FFT stuff
         mrateHi = [mrateHi; fftrf{ex}.(field)(cc).rateHi];
         mrateLow = [mrateLow; fftrf{ex}.(field)(cc).rateLow];
         fftcorr = [fftcorr; fftrf{ex}.(field)(cc).corrrho];
         
        fftpBY = [fftpBY; benjaminiYekutieli(fftrf{ex}.(field)(cc).corrp, 0.05)];
        fftpBH = [fftpBH; benjaminiHochbergFDR(fftrf{ex}.(field)(cc).corrp, 0.05)];
        fftpval = [fftpval; fftrf{ex}.(field)(cc).corrp];
        
        sft = fftrf{ex}.rfs_post(cc).sftuning.rhos'*fftrf{ex}.rfs_post(cc).sftuning.B;
        sftuningE = [sftuningE; sft(:,1)'];
        sftuningL = [sftuningL; sft(:,2)'];
        
        sphi = fftrf{ex}.rfs_post(cc).sftuning.sfpsthhi(:,sfix);
        splow = fftrf{ex}.rfs_post(cc).sftuning.sfpsthlow(:,sfix);
        
        spsthhi1 = [spsthhi1; sphi(:,1)']; 
        spsthhi10 = [spsthhi10; sphi(:,2)']; 
        spsthlow1 = [spsthlow1; splow(:,1)']; 
        spsthlow10 = [spsthlow10; splow(:,2)']; 

         % fixrate
         nipeakt = [nipeakt; fixrat{ex}.BackImage.peakloc(cc)];
         nipeakv = [nipeakv; fixrat{ex}.BackImage.peak(cc)];
         nitrot = [nitrot; fixrat{ex}.BackImage.troughloc(cc)];
         nitrov = [nitrov; fixrat{ex}.BackImage.trough(cc)];
         nimrate = [nimrate; fixrat{ex}.BackImage.meanRate(cc,:)]; % mean rate
         grpeakt = [grpeakt; fixrat{ex}.Grating.peakloc(cc)];
         grpeakv = [grpeakv; fixrat{ex}.Grating.peak(cc)];
         grtrot = [grtrot; fixrat{ex}.Grating.troughloc(cc)];
         grtrov = [grtrov; fixrat{ex}.Grating.trough(cc)];
         grmrate = [grmrate; fixrat{ex}.Grating.meanRate(cc,:)];
        
         wf = [wf; Waveforms{ex}(cc)];
         % Counter
         ctr = [ctr; [numel(r2) numel(gtr2) size(mrateHi,1)]];
         
         if ctr(end,1) ~= ctr(end,3)
             keyboard
         end
    end
end

% wrap orientation
oriPref(oriPref < 0) = 180 + oriPref(oriPref < 0);
oriPref(oriPref > 180) = oriPref(oriPref > 180) - 180;

fprintf('%d (Spatial) and %d (Grating) of %d Units Total are significant\n', sum(sigs), sum(sigg), numel(sigs))

cg = arrayfun(@(x) x.cg, wf);
ecrl = arrayfun(@(x) x.ExtremityCiRatio(1), wf);
ecru = arrayfun(@(x) x.ExtremityCiRatio(2), wf);
wfamp = arrayfun(@(x) x.peakval - x.troughval, wf);

%%

ix = sigg & sigs;
ix = wfamp > 40;
ix = ix & ecc < .5;

% ix = ix & max(sftuningE,[],2) > .05;
figure(1); clf
tl = mean(sftuningE(:,1:2),2) > mean(sftuningE(:,5:end),2);
ixE = ix & tl;
% ixL = ix & max(sftuningL(:,4:end),[],2)>.05;
tl = mean(sftuningE(:,1:2),2) < mean(sftuningE(:,5:end),2);
ixL = ix & tl;

m = spsthhi1(ixL,:); m = m ./ max(median(m,2), 1);
plot(lags, mean(m), 'b', 'Linewidth', 2); hold on
m = spsthlow1(ixL,:); m = m ./ max(median(m,2), 1);
plot(lags, mean(m), 'b--', 'Linewidth', 2);

m = spsthhi10(ixL,:); m = m ./ max(median(m,2), 1);
plot(lags, mean(m), 'r', 'Linewidth', 2);
m = spsthlow10(ixL,:); m = m ./ max(median(m,2), 1);
plot(lags, mean(m), 'r--', 'Linewidth', 2);
xlim([-0.05 .3])

figure(2); clf
plot(sfs, sftuningE(ixE,:)', 'b'); hold on
plot(sfs, sftuningE(ixL,:)', 'r')


sdiff = spsthhi1-spsthlow1;
clf
plot(lags, mean(sdiff(ixE,:))); hold on
sdiff = spsthhi10-spsthlow10;
plot(lags, mean(sdiff(ixL,:)));
xlim([-0.05 .3])


