function rf = get_rf_sig(X, Y, numlags, inds, dims, varargin)
% rf = get_rf_sig(X, Y, numlags, inds, dims, varargin)
% ip.addParameter('numboot', 100)
% ip.addParameter('plot', false)
% ip.addParameter('thresh', .1)
% ip.addParameter('smoothing', 0)

ip = inputParser();
ip.addParameter('numboot', 100)
ip.addParameter('plot', false)
ip.addParameter('thresh', .1)
ip.addParameter('smoothing', 0)
ip.parse(varargin{:})

if isempty(inds)
    inds = 1:size(X,1);
end

inds = inds(:);

%%
numboot = ip.Results.numboot;
% negative time lags
[stasNull,Nnull] = simpleForcorrValid(X, Y, numboot, inds, -numboot);

% positive time lags
[stasFull, Nstim] = simpleForcorrValid(X, Y, numlags, inds, 0);

%%
ss = std(stasFull);
sn = std(stasNull);
ssrat = squeeze(ss ./ sn);

[~, iix] = sort(max(ssrat));
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(10); clf
ax = plot.tight_subplot(sx, sy, 0, 0);
for i = 1:NC
    set(gcf, 'currentaxes', ax(i));
    cc = iix(i);
    imagesc(reshape(ssrat(:,cc), dims), [0 3])
    hold on
    plot([0 dims(2)], [0 0], 'w')
    plot([0 dims(2)], [dims(1), dims(1)], 'w')
    plot([dims(2) dims(2)], [0, dims(1)], 'w')
    plot([0 0], [0, dims(1)], 'w')

    axis off
end

figure(12); clf
ax = plot.tight_subplot(sx, sy, 0, 0);
for i = 1:NC
    set(gcf, 'currentaxes', ax(i));
    cc = iix(i);
    imagesc(reshape(ssrat(:,cc), dims), [2 inf]); 
    hold on
    plot([0 dims(2)], [0 0], 'w')
    plot([0 dims(2)], [dims(1), dims(1)], 'w')
    plot([dims(2) dims(2)], [0, dims(1)], 'w')
    plot([0 0], [0, dims(1)], 'w')

    axis off
end

figure(11); clf
ax = plot.tight_subplot(sx, sy, 0, 0);
for i = 1:NC
    set(gcf, 'currentaxes', ax(i));
    cc = iix(i);
    plot(wf(cc).lags, wf(cc).isi)
    axis off
end
%%

%%
if ~exist('cc', 'var')
    cc = 0;
end

cc = cc + 1;
if cc > NC
    cc = 1;
end



sss = zeros(NC,1);
for cc = 1:NC
figure(6); clf;
null = stasNull(:,:,cc);
% null = Ymu(:,:,cc);
sta = stasFull(:,:,cc);
mn = min(null);
mx = max(null);

% mn = min(min(null(:)), min(sta(:)));
% mx = max(max(null(:)), max(sta(:)));

subplot(1,3,1)
imagesc(null)
% imagesc(null, [mn mx])
subplot(1,3,2)
imagesc(sta)
% , [mn mx]);
subplot(1,3,3)
imagesc( (sta > mx) -  (sta < mn) )

figure(2); clf
subplot(1,3,1)
sn = reshape(std(stasNull(:,:,cc)), dims);
imagesc(sn)
colorbar
subplot(1,3,2)
ss = reshape(std(stasFull(:,:,cc)), dims);
imagesc(ss)
colorbar
subplot(1,3,3)
imagesc(ss./sn)
title(cc)
colorbar
drawnow
sss(cc) = max(ss(:)./sn(:));

figure(3); clf
subplot(1,2,1)
plot(wf(cc).waveform)
subplot(1,2,2)
plot(wf(cc).lags, wf(cc).isi)
title(cc)
end

%%
ss = std(stasFull);
sn = std(stasNull);
ssrat = squeeze(ss ./ sn);

[~, iix] = sort(sss);
sx = ceil(sqrt(NC));
sy = round(sqrt(NC));
figure(10); clf
ax = plot.tight_subplot(sx, sy, 0, 0);
for i = 1:NC
    set(gcf, 'currentaxes', ax(i));
    cc = iix(i);
    imagesc(reshape(ssrat(:,cc), dims), [0 3])
    axis off
end

figure(11); clf
ax = plot.tight_subplot(sx, sy, 0, 0);
for i = 1:NC
    set(gcf, 'currentaxes', ax(i));
    cc = iix(i);
    plot(wf(cc).lags, wf(cc).isi)
    axis off
end

%%

%% Loop over cells and check significance
sm = ip.Results.smoothing;

debugit = true;

rf = struct();

rf.stasRaw = stasFull;
rf.stasNorm = zeros(size(stasFull));
rf.ci = zeros(2,NC);
rf.volume = zeros(NC,1);
rf.sigrf = zeros(NC,1);
rf.centroids = cell(NC,1);

for cc = 1%:NC

    % get null distribution
    nullrat = Nstim ./ Nnull; % scale by difference in N
    null = stasNull(:,:,cc);

    munull = mean(null);
    sdnull = std(null);

%     nmask = sdnull./max(sdnull(:));
    nmask = Nstim ./ max(Nstim(:));
    
    if sm > 0
        munull = reshape(imgaussfilt(reshape(munull, dims), sm), [], 1)';
        sdnull = reshape(imgaussfilt(reshape(sdnull, dims), sm), [], 1)';
    end

    null = stasNull(:,:,cc);
    null = (null - munull) ./ sdnull;
    null = null .* nullrat;
    
    sta = stasFull(:,:,cc);
    sta = (sta - munull) ./ sdnull;
    sta = sta.*nmask;

    rf.stasNorm(:,:,cc) = sta;
    thresh = ip.Results.thresh / 2 / numlags;
%     thresh = ip.Results.thresh / 2 / size(X,2);
% ci = prctile(null, [thresh 100-thresh], 1);

    ci = prctile(null(:), [thresh 100-thresh], 1);
    ci = ci.*ones(1,size(null,2));
    
    rf.ci(1,cc) = ci(1,1);
    rf.ci(2,cc) = ci(2,1);

    figure(6); clf;
    null = stasNull(:,:,cc);
    sta = stasFull(:,:,cc);
    mn = min(min(null(:)), min(sta(:)));
    mx = max(max(null(:)), max(sta(:)));

    subplot(1,3,1)
    imagesc(null, [mn mx])
    subplot(1,3,2)
    imagesc(sta, [mn mx]);

    %%
    mn = min(null(:));
    mx = max(null(:));

    
    % smooth confidence intervals
    % ci(1,:) = reshape(imgaussfilt(reshape(ci(1,:), dims), sm), [], 1)';
    % ci(2,:) = reshape(imgaussfilt(reshape(ci(2,:), dims), sm), [], 1)';
    
    sigrf = sta > ci(2,:) | sta < ci(1,:);

    if ip.Results.plot
        figure(1); clf;
        subplot(2,1,1)

        plot(max(sta), 'k')
        hold on
        plot(min(sta), 'k')
        plot(ci', 'Color', ones(1,3)*.5, 'Linewidth', 1)

        subplot(2,1,2)
        
        plot(mean(sigrf), 'k')
        hold on
        
        title(mean(sigrf(:)))
        drawnow
    end


    sigrf = reshape(sigrf, [numlags, dims]); % 3D spatiotemporal tensor
    rf.sigrf(cc) = mean(sigrf(:));

    bw = bwlabeln(sigrf);
    s = regionprops3(bw);
    if ~isempty(s)
        [~, id] = max(s.Volume);
        s = s(id,:);
        rf.volume(cc) = s.Volume;
        rf.centroids{cc} = s.Centroid;
    end
%     s(s.Volume < 2,:) = []; % remove singular significant pixels
end
%     sigrf = mean(sigrf(:));

