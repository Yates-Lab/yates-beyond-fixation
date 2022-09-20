function polar_grid(th, rho, varargin)
% plot a polar grid overlay
% polar_grid(theta, rho)
% Inputs:
%   Theta <double> grid steps for orientation (in degrees)
%   Rho   <double> grid steps for eccentricity
% Optional Arguments:
%   Color           color of grid
%   TextColor       color of text labels
%   TextOffset      distance from grid

ip = inputParser();
ip.addParameter('Color', [.5 .5 .5])
ip.addParameter('TextColor', [.2 .2 .2])
ip.addParameter('TextOffset', .5)
ip.addParameter('RhoLabelTheta', 0)
ip.parse(varargin{:})

grid_spacing{1} = th(:)';
grid_spacing{2} = rho(:)';

mrho = max(grid_spacing{2});
to = ip.Results.TextOffset; % text offset
for th = grid_spacing{1}
    tx = mrho*cosd(th);
    ty = mrho*sind(th);
    plot([0 tx], [0 ty], 'Color', ip.Results.Color); hold on
    if ~all(ip.Results.TextColor==1)
        text(tx + to*cosd(th)-to/2, ty + to*sind(th), num2str(th), 'Color', ip.Results.TextColor)
    end
end

th = 0:max(grid_spacing{1});
th0 = ip.Results.RhoLabelTheta;
for rho = grid_spacing{2}
    plot(rho.*cosd(th), rho.*sind(th), 'Color', ip.Results.Color);
    if ~all(ip.Results.TextColor==1)
        text(rho.*cosd(th0) + to*cosd(th0), rho.*sind(th0) + to*sind(th0) - to/2, num2str(rho), 'Color', ip.Results.TextColor) 
    end
end

axis off

