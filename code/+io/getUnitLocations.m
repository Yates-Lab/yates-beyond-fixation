function S = getUnitLocations(osp, pow)
% S = getUnitLocations(osp)
% Input:
%   osp [struct] - getSpikesFromKilo output
%   pow [double] - soft max power    
% Output:
%   S [struct]
%       x 
%       y 
%       templates
debug = false;

if nargin < 2
    pow = 20;
end

NC = numel(osp.cids);

x = nan(NC,1);
y = nan(NC,1);

if isfield(osp, 'WFmed')
    useWF = true;
    temps = osp.WFmed;
else
    useWF = false;
    temps = osp.temps;
end

nLags = size(temps,2);
nChan = size(temps,3);

if nChan == 64
    if numel(osp.xcoords)~=nChan
        warning('hack channel map for 2-shank probes')
        osp.xcoords = [zeros(32, 1); 200*ones(32, 1)];
        osp.ycoords = [(1:32)'*35; (1:32)'*35];
    end
end

xcoords = osp.xcoords;
ycoords = osp.ycoords;
    
templates = zeros(nLags, nChan, NC);
for cc = 1:NC
    if useWF
        unitTemp = squeeze(temps(cc,:,:));
    else    
        tempix = unique(osp.spikeTemplates(osp.clu==osp.cids(cc)))+1;
        unitTemp = squeeze(mean(osp.tempsUnW(tempix, :, :),1));
    end
    templates(:,:,cc) = unitTemp;
    
    u = sum(unitTemp.^2,1);
    w = u.^pow /sum(u(:).^pow);
    
    x(cc) = w*xcoords;
    y(cc) = w*ycoords;
    
    if debug
        figure(1); clf
        for ch = 1:nChan
            plot(xcoords(ch)/100 + osp.WFtax, ycoords(ch) + unitTemp(:,ch), 'k', 'Linewidth', 2); hold on
        end
        
        shctrs = unique(xcoords);
        unorm = u/max(u);
        for sh = 1:numel(shctrs)
            shix = xcoords==shctrs(sh);
            plot(mean(xcoords(shix)/100) + unorm(shix), ycoords(shix), 'r', 'Linewidth', 2)
        end
        
        plot(xlim, y(cc)*[1 1], 'r--', 'Linewidth', 2)
        plot(x(cc)/100*[1 1], ylim, 'r--', 'Linewidth', 2)
        title(sprintf('Unit %d', cc))
        ylabel('Depth')
        input('Check')
        
    end
end

S.x = x;
S.y = y;
S.templates = templates;
if useWF
    S.ciHi = permute(osp.WFciHi, [2 3 1]);
    S.ciLow = permute(osp.WFciLow, [2 3 1]);
end
S.xcoords = xcoords;
S.ycoords = ycoords;
S.useWF = useWF;