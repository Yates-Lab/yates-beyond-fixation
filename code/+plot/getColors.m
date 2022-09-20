function clrs = getColors(C, cmin, cmax, cmap)
% clrs = getColors(C, cmin, cmax)
    
    
    if nargin < 2 || isnan(cmin)
        % diverging colorscheme, so take abs        
        cmin = min(C);
    end
    
    if nargin < 3 || isnan(cmax)
        cmax = max(C);
    end
    
    if nargin < 4
        cmap = colormap;
    end
    m = size(cmap,1);
    if cmax == cmin
        disp('cmax = cmin')
        cind = round(m/2)*ones(size(C,1),1);
    else
        cind = fix((C-cmin)/(cmax-cmin)*m)+1;
    end
    cind(cind<1) = 1;
    cind(cind>m) = m;
    clrs = cmap(cind,:);
end