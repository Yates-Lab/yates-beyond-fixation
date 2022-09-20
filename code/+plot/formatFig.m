function [] = formatFig(hFig, figSz, style, varargin)
%   formatFig(hFig, figSz, style, varargin)
%
% Format figure to specific font name, size, etc. 
% Applies changes to all subplots within (skips legends).
% Enables multiple styles e.g. nature, jnsci etc.. 
%
% INPUT:
%   hFig     - figure handle (defualt is current figure)
%   figSz    - figure size in inches (default is [3 3])
%   style    - nature, jnsci, etc. see switch loop in code. modular. 
%   varargin - input here overrides 'style' settings (e.g. 'LineWidth', 4) 

% 20160105 lnk wrote it

%% set defualts:
if ~exist('hFig', 'var')
    hFig = gcf;
end
figure(hFig);
if ~exist('figSz', 'var') || isempty(figSz)
    figSz = get(hFig, 'PaperSize');
end
if ~exist('style', 'var')
    style = 'default';
end

% other options:
% set(gca,'ticklength',[0.01 0.025])

%% style specific:
switch style
    case 'nature'
        FontSizeTitle   = 6;
        FontSizeLabelX  = 5;
        FontSizeLabelY  = 5;
        FontSizeLabelZ  = 5;
        FontSizeAxes    = 5;
        LineWidth       = .5; 
        FontName        = 'Helvetica';
        Box             = 'off';
        TickDir         = 'out';
        
    case 'jnsci'
        FontSizeTitle   = 8;
        FontSizeLabelX  = 7;
        FontSizeLabelY  = 7;
        FontSizeLabelZ  = 7;
        FontSizeAxes    = 7;
        LineWidth       = .5;
        FontName        = 'Helvetica';
        Box             = 'off';
        TickDir         = 'out';
        
    case 'eneuro'
        FontSizeTitle   = 8;
        FontSizeLabelX  = 7;
        FontSizeLabelY  = 7;
        FontSizeLabelZ  = 7;
        FontSizeAxes    = 7;
        LineWidth       = .5;
        FontName        = 'Helvetica';
        Box             = 'off';
        TickDir         = 'out';
        
    case 'poster'
        FontSizeTitle   = 30;
        FontSizeLabelX  = 25;
        FontSizeLabelY  = 25;
        FontSizeLabelZ  = 25;
        FontSizeAxes    = 25;
        LineWidth       = 2; 
        FontName        = 'Helvetica';
        Box             = 'off';
        TickDir         = 'out';
        
    
        
    case 'default'
        FontSizeTitle   = 12;
        FontSizeLabelX  = 10;
        FontSizeLabelY  = 10;
        FontSizeLabelZ  = 10;
        FontSizeAxes    = 8;
        LineWidth       = 1;
        FontName        = 'Helvetica';
        Box             = 'off';
        TickDir         = 'out';
    
    otherwise 
        error('Your selected ''style'' is invalid. Either correct or leave blank for ''defualt''');
end


%% check varargin:

% if none exist take the 'style' settings
p = inputParser();
p.addOptional('FontSizeTitle', FontSizeTitle)
p.addOptional('FontSizeLabelX', FontSizeLabelX)
p.addOptional('FontSizeLabelY', FontSizeLabelY)
p.addOptional('FontSizeLabelZ', FontSizeLabelZ)
p.addOptional('FontSizeAxes', FontSizeAxes);
p.addOptional('LineWidth', LineWidth);
p.addOptional('FontName', FontName);
p.addOptional('Box', Box);
p.addOptional('TickDir', TickDir);
p.addOptional('pretty', false);
p.addOptional('OffsetAxesBool', false)
p.parse(varargin{:})
            
%% Set format:

set(hFig, 'PaperSize', figSz, 'PaperPosition', [0 0 figSz]);

% identify legends and turn off boxes:
hl  = findobj(gcf,'Type','legend');
set(hl, 'Box', 'off')
set(hl, 'FontSize', p.Results.FontSizeAxes);
set(hl,'FontName', p.Results.FontName, 'color', 'none')

% identify all subplots ('axes') and format:

ha = findobj(gcf,'Type','axes');
    
set(ha, 'Box', p.Results.Box)
set(ha, 'TickDir', p.Results.TickDir)
set(ha, 'LineWidth', p.Results.LineWidth)
set(ha, 'FontSize', p.Results.FontSizeAxes);
% set(ha,'FontName', p.Results.FontName, 'color', 'none')

for ii = 1:numel(ha)
    ht = get(ha(ii),'title');
    set(ht,'FontName', p.Results.FontName, 'fontweight', 'normal', 'FontSize', p.Results.FontSizeTitle);
    hx = get(ha(ii),'xlabel');
    set(hx,'FontName', p.Results.FontName, 'fontweight', 'light', 'FontSize', p.Results.FontSizeLabelX);
    hy = get(ha(ii),'ylabel');
    set(hy,'FontName', p.Results.FontName, 'fontweight', 'light', 'FontSize', p.Results.FontSizeLabelY);
    hz = get(ha(ii),'zlabel');
    set(hz,'FontName', p.Results.FontName, 'fontweight', 'light', 'FontSize', p.Results.FontSizeLabelZ);
    
    % set xy axes color to black:
    set(ha(ii), 'XColor', [0 0 0]);
    set(ha(ii), 'YColor', [0 0 0]);
    set(ha(ii), 'ZColor', [0 0 0]);
    set(ha(ii), 'Layer', 'top');
    set(ha(ii), 'TickLength', [0.025 0.05])
    
    tex = findobj(ha(ii), 'Type', 'Text');
    for jj = 1:numel(tex)
        set(tex(jj),'FontName', p.Results.FontName, 'fontweight', 'light', 'FontSize', p.Results.FontSizeAxes);
    end
    
    if p.Results.OffsetAxesBool
        plot.offsetAxes(gca);
    end
end

htla = findobj(gcf, 'Type', 'TiledLayout');
if ~isempty(htla)
    htla.Title.FontSize = p.Results.FontSizeAxes;
    htla.XLabel.FontSize = p.Results.FontSizeAxes;
    htla.YLabel.FontSize = p.Results.FontSizeAxes;
    htla.Title.FontName = p.Results.FontName;
    htla.XLabel.FontName = p.Results.FontName;
    htla.YLabel.FontName = p.Results.FontName;
    htla.Title.Color = [0 0 0];
    htla.XLabel.Color = [0 0 0];
    htla.YLabel.Color = [0 0 0];
    
end

if p.Results.pretty
    ha.Color        = [.95 .95 .95];
    grid on
    ha.GridAlpha    = .5;
    ha.GridColor    = [1 1 1];
end

% On some rare cases, the (default) opengl renderer will save bitmaps. 
% This ensures we don't lose the vectorization:
set(gcf, 'renderer', 'painters')


