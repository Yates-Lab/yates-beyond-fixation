function Basis = raised_cosine(x, n, b, pows)
% Basis = raised_cosine(x, n, b, pows)

testing = 0;

if numel(n)==1
    ctrs = 1:n;
else
    ctrs = n;
end

xnl = nl(x,b,pows);
xnl(isinf(xnl)) = 0;

if testing
    figure(1); clf
    subplot(2,2,1)
    plot(x, xnl)
end

xdiff = abs(xnl-ctrs);

if testing
    subplot(2,2,2)
    plot(xdiff)
end

Basis = cos(max(-pi, min(pi, xdiff*pi)))/2 + .5;

function y = nl(x, b, pows)
y = log(x/(b+1e-20))/log(pows) + 1;

function x = invnl(y,b,pows)
x = b*pows.^(y-1);


