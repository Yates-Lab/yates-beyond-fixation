function plotWaveforms(W, scale, varargin)
    
    if nargin < 2
        scale = 1;
    end
    
    if ~isfield(W, 'waveform') && isfield(W, 'osp')
        W = io.get_waveform_stats(W.osp);
    end


%     figure; clf
    NC = numel(W);
    ip = inputParser();
    ip.addParameter('cmap',lines(NC))
    ip.addParameter('overloadSU', false)
    ip.addParameter('cids', []);
    ip.parse(varargin{:})
    
    cmap = ip.Results.cmap;
    
    if isempty(ip.Results.cids)
        cids = 1:NC;
    else
        cids = ip.Results.cids;
        NC = numel(cids);
    end
    
    SU = false(NC,1);
    
    for cc = cids(:)'
        nw = norm(W(cc).waveform(:,3));
%         SU(cc) = nw > 20 & W(cc).isiL > .1 & W(cc).isiL < 1 & W(cc).uQ > 5 & W(cc).isi(200) < 0;
        if ip.Results.overloadSU
            SU(cc) = true;
        else
            SU(cc) = W(cc).isiV < .1;
        end
        
        nts = size(W(1).waveform,1);
        xax = linspace(0, 1, nts) + cc;
        if SU(cc)
            clr = cmap(cc,:);
        else
            clr = .5*[1 1 1];
        end
        plot(xax, W(cc).waveform*scale + W(cc).spacing - W(cc).depth, 'Color', clr, 'Linewidth', 2); hold on
        text(mean(xax),  W(cc).spacing(end)+20 - W(cc).depth, sprintf('%d', cc))
        
    end
    
    xlabel('Unit #')
    ylabel('Depth')
    plot.fixfigure(gcf, 12, [8 4], 'OffsetAxes', false)