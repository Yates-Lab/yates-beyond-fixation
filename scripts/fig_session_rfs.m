user = 'jakelaptop';
addFreeViewingPaths(user);
addpath Analysis/manuscript_freeviewingmethods/
figDir = 'Figures/manuscript_freeviewing/fig02';

%% Load Data

sorter = 'kilowf'; % id tag for using kilosort spike-sorting

clear Srf

sesslist = io.dataFactory;
sesslist = sesslist(1:57);

sfname = fullfile('Data', 'spatialrfsreg.mat');
Srf = cell(numel(sesslist),1);

assert(exist(sfname, 'file')==2, 'run fig02_spatRF_gratRF first')
disp('Loading Spatial RFs')
load(sfname)
disp('Done')


%%
for ex = 39:numel(sesslist)
    
    if isempty(Srf{ex}) || ~isfield(Srf{ex}, 'coarse')
        continue
    end
    figure(1); clf
    NC = numel(Srf{ex}.coarse.rffit);
    sx = ceil(sqrt(NC));
    sy = round(sqrt(NC));
    ax = plot.tight_subplot(sx,sy, .01, 0.01);
    for cc = 1:(sx*sy)
        if cc > NC
            axis(ax(cc), 'off')
            continue
        end
        
        set(gcf, 'currentaxes', ax(cc));
        I = Srf{ex}.coarse.srf(:,:,cc);
%         I = (I - min(I(:))) / (max(I(:)) - min(I(:)));
        imagesc(Srf{ex}.coarse.xax, Srf{ex}.coarse.yax, I);
        hold on
        if isfield(Srf{ex}, 'fine')
            I = Srf{ex}.fine.srf(:,:,cc);
%             I = (I - min(I(:))) / (max(I(:)) - min(I(:)));
            imagesc(Srf{ex}.fine.xax, Srf{ex}.fine.yax, I);
        end
        
        [xx, yy] = meshgrid(-15:5:15);
        clr = .2*[1 1 1];
        plot(xx, yy, 'Color', clr)
        plot(xx', yy', 'Color', clr)
        axis xy
        axis off
        
        text(-12, 7, sprintf('%d', cc))
        
    end
    
    colormap(plot.coolwarm)
    plot.formatFig(gcf, [sx sy], 'default')
    saveas(gcf, fullfile(figDir, sprintf('rfs_%s.pdf', sesslist{ex})))
end