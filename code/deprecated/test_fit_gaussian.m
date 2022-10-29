
%% test fit gaussian


[xx,yy] = meshgrid(-20:.5:20);

dims = size(xx);
mu = [2.1; -1];
C = [.5 -.1;-.1 .5];
Cinv = pinv(C);

X = [xx(:) yy(:)]';

I = exp(-.5 * sum( ((X - mu)'*Cinv).*(X - mu)' ,2) );


figure(1); clf;
imagesc(xx(1,:), yy(:,1)', reshape(I, dims));

[con, ar, ctr, thresh, maxoutrf] = get_rf_contour(xx, yy, reshape(I, dims));

fit = fit2Dgaussian(xx, yy, reshape(I, dims), [ctr ar 0 0]);

hold on
plot.plotellipse(fit.mu, fit.C, 1, 'Color', 'r');
plot(con(:,1), con(:,2), 'y')

% sc = sqrt(-log(.5))*2;

[u, s] = svd(fit.C);
s = sqrt(s);
arfit = prod(diag(s))*pi;


disp([ar, arfit])