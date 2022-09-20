function [] = fixfigure(figurehandle, fontSize, paperSize, varargin)
% Fix a matlab figure to be publication ready
% fixfigure(figurehandle, fontSize, paperSize, varargin)
% Optional Arguments:
%   'TickDir' (default: 'out')
%   'FontName' (default: 'Helvetica')
%   'FontWeight' (default: 'normal')
%   'Linewidth' (default: .5)
%   'OffsetAxes' (default: true) % Axes don't come together

% defaults
if nargin < 3
    paperSize = [6 3.5];
    if nargin < 2
        fontSize = 12;
    end
end

% options (not implemented yet)
p=inputParser();
p.addOptional('TickDir', 'out');
p.addOptional('FontName', 'Helvetica');
p.addOptional('FontWeight','normal');
p.addOptional('LineWidth',.5);
p.addOptional('FontSize', fontSize);
p.addOptional('OffsetAxes', true);
p.parse(varargin{:})
opts=p.Results;

axisOptions= fieldnames(opts);

% Check to see if a specific figure is identified for
% Otherwise the current figure is used:
if exist('figurehandle','var')
    set(0,'currentfigure',figurehandle);
end

figureChildren = get(gcf,'children');

fixChildren(figureChildren,axisOptions,opts)

set(gcf, 'Papersize', paperSize, 'paperposition', [0 0 paperSize])
set(gcf, 'Color', 'w')

set(gcf, 'Renderer', 'painters')


function fixChildren(figureChildren,axisOptions,opts)
% Loop through all the children of the figure:
for kChild = 1:length(figureChildren)
    if isa(figureChildren(kChild), 'matlab.graphics.layout.TiledChartLayout')
        fixChildren(get(figureChildren(kChild), 'children'), axisOptions, opts);
    else
        currentChildProperties = get(figureChildren(kChild));
        if isfield(currentChildProperties, 'ActivePositionProperty')
            set(gcf,'currentaxes',figureChildren(kChild))
            set(gca, 'YColor', [0 0 0], 'XColor', [0 0 0], 'ZColor', [0 0 0], 'Layer', 'top')
            set(gca, 'TickLength', [0.025 0.05])
            axisChildren = get(gca,'children');
            currentAxisProperties = get(gca);
            for ii = 1:numel(axisOptions)
                if isfield(currentAxisProperties, axisOptions{ii}) && strcmpi(currentAxisProperties.Type, 'axes')
                    set(figureChildren(kChild), axisOptions{ii}, opts.(axisOptions{ii}));
                end
            end
            
            % And loop through all the children of each axis:
            for kAxis = 1:length(axisChildren)
                axisfields = get(axisChildren(kAxis));
                
                % Loop through axis options and modify the axis
                for ii = 1:numel(axisOptions)
                    if isfield(axisfields, axisOptions{ii}) && ~strcmpi(axisOptions{ii}, 'Linewidth')
                        
                        set(axisChildren(kAxis), axisOptions{ii}, opts.(axisOptions{ii}));
                    end
                end
                
            end
            
            ht = get(gca,'title');
            set(ht,'FontName',opts.FontName,'FontSize',opts.FontSize, 'Color', 'k');
            hx = get(gca,'xlabel');
            set(hx,'FontName',opts.FontName, 'FontWeight', opts.FontWeight,'FontSize',opts.FontSize, 'Color', 'k');
            hy = get(gca,'ylabel');
            set(hy,'FontName',opts.FontName, 'FontWeight', opts.FontWeight,'FontSize',opts.FontSize, 'Color', 'k');
            hl  = findobj(gcf,'Type','axes','Tag','legend');
            set(hl,'box','off')
        end
        
        if opts.OffsetAxes
            plot.offsetAxes(gca)
        end
        
        box off
    end
    
end

