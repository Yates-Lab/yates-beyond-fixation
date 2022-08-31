%% add paths
user = 'jakelaptop';
addFreeViewingPaths(user);
addpath Analysis/manuscript_freeviewingmethods/

%% If you want to recompute in matlab using shifter
flist = dir('Data/*.hdf5');

fi = 1;
splits = regexp(flist(fi).name, '_', 'split');
s.subject = splits{1};
s.date = splits{2};
s.rect = cellfun(@str2double, splits(3:6));
s.smoothing = str2double(splits{8});

fname = fullfile(flist(fi).folder, flist(fi).name);
finf = h5info(fname);
stimfields = arrayfun(@(x) x.Name(2:end), finf.Groups, 'uni', 0);
stimfields = setdiff(stimfields, 'Neurons');

%% loop over stimuli

Fstat = struct();

win = hanning(dims(2))*hanning(dims(3))';

%Compute autocorrelation


nxfft = 2^nextpow2(dims(3));
nyfft = 2^nextpow2(dims(2));

fprintf('Looping over stimulus conditions\n')
for istim = 1:numel(stimfields)
    stim = stimfields{istim};
    tset = 'Train';
    ppd = h5readatt(fname, ['/' stim '/' tset '/Stim'], 'ppd');
    
    fprintf("Stimulus: %s\n", stim)
    
    Stim = h5read(fname, ['/' stim '/' tset '/Stim']);

    dims = size(Stim);
    dims(1) = 10e3;
    Fstat.(stim).Fpow = zeros([dims(1) nyfft nxfft]);
    Fstat.(stim).Fphase = zeros([dims(1) nyfft nxfft]);
    
    for iFrame = 1:dims(1)
    
        I = double(squeeze(Stim(iFrame,:,:)))/127;
%         I = I.*win;
%         Corrxx = xcorr2(I, I);
        
        F = fftshift(fft2(I, nxfft, nyfft));
        F1 = fftshift(fft2(I));
        
    
        Fstat.(stim).Fpow(iFrame,:,:) = abs(F);
        Fstat.(stim).Fphase(iFrame,:,:) = angle(F);
        Fstat.(stim).ppd = ppd;
        Fstat.(stim).xax = xax;
    end
end

%%
Fs = Fstat.(stim).ppd;
N = dims(2)*dims(3);

xax = linspace(-1, 1, size(Fstat.(stim).Fpow,2))*Fs/2;

n = numel(stimfields);

figure(1); clf
figure(2); clf
for istim = 1:n
    stim = stimfields{istim};
    
    
    figure(1)
    subplot(1, n, istim)
    Idft = squeeze(mean(Fstat.(stim).Fpow,1));
    
    Ipsd = 2 * Idft.^2 / (Fs*N);
    
    imagesc(xax, xax, log10(Ipsd)/2);
    
    title(stim)
    
    figure(2)
    xd = xax(xax>0);
    Pxx = radialavg(Ipsd, xax, xd, 0)/2;
%     Pxx = Ipsd(floor(size(Ipsd,1)),xax>0);
    
    plot(xd, log10(Pxx)); hold on
    
    
%     plot(xax(xax>0), log10(Ipsd(36,xax>0))); hold on
end

% oof = -2*log10(xd);
% plot(xd, oof - min(oof) + -4, 'k--')


legend({'Images', 'Sparse Noise', 'Gabor Noise'})
set(gca, 'xscale', 'log')

set(gca, 'XTick', [1 2 5 10 20])
xlim([0.5 20])

ylabel('Log Power')
xlabel('Frequency (cyc/deg)')

plot.formatFig(gcf, [5 5], 'default')
saveas(gcf, 'Figures/manuscript_freeviewing/stimpowerspectrum.pdf')