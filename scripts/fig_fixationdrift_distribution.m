
%% Loop over sessions with DDPI and get the drift distribution
sesslist = io.dataFactory;
sessIds = find(contains(sesslist, 'logan'));
nSessions = numel(sessIds);
D = cell(nSessions,1);
for isess = 1:nSessions
    Exp = io.dataFactory(sessIds(isess), 'spike_sorting', 'kilowf');
    D{isess} = get_drift_distribution(Exp);
end
    

%% Plot the distribution after averaging across sessions

ddist = 0;
for i = 1:nSessions
    dtmp = D{isess};
    dtmp = imgaussfilt(dtmp, 1);
    dtmp(dtmp <= 1) = 0;
    ddist = ddist + dtmp;
end

ddist = ddist ./ sum(ddist,2);
ddist = imgaussfilt(ddist,1);
ddist = log10(ddist);
ddist(ddist < -3) = -inf;
figure(1); clf
contourf(ddist, 50, 'Linestyle', 'none')
% imagesc(ddist)
c = colorbar;
c.Limits = [-5 0];
c.Ticks = -5:1:0;
axis xy
colormap parula
ylim([0 500])

xlabel('Arcminutes')
ylabel('Fixation Time (ms)')

plot.formatFig(gcf, [1 1], 'nature')
saveas(gcf, 'Figures/2021_ucbtalk/driftdist.pdf')