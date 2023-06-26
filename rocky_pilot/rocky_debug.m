

%%

Exp = load('~/Downloads/r20230608_dpi.mat/');

sp = load('~/Downloads/spkilo.mat');

%%

EyeX = Exp.vpx.smo(:,2);
EyeY = Exp.vpx.smo(:,3);
figure(1); clf
plot(EyeX, '-'); hold on
plot(xlim, [0 0])
% plot(EyeY, '-')


xax = linspace(-15, 15, 50);
n = numel(EyeX);
s = floor(n/2);
C = histcounts2(EyeX(s:end), EyeY(s:end), xax, xax);

figure(2); clf
imagesc(xax, xax, log(C'))
colormap parula
xlabel('horizontal position')
ylabel('vertical position')

%%
inds = io.getValidTrials(Exp, 'FixRsvpStim');
i = inds(1);
figure(1); clf
% plot(EyeX, '-'); hold on
% plot(xlim, [0 0])
% plot(EyeY, '-')
for i = inds(:)'
plot(Exp.D{i}.PR.NoiseHistory(:,1), Exp.D{i}.PR.NoiseHistory(:,4)*.1-5, 'r'); hold on
end
plot(Exp.vpx.smo(:,1), EyeX, 'k')
%%

% Exp.D{i}.PR.