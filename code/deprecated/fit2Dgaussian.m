function stat = fit2Dgaussian(xx, yy, I, par0)
% fit2Dgaussian(xx, yy, I, par0)
% xx 
% yy
% par0 = initial parameter guess
% [mux, muy, CDiag, COff, noiseFloor]

X = [xx(:) yy(:)];

mxI = max(I(:));
mnI = min(I(:));
% gaussian function
gfun = @(params, X) params(5) + (mxI - params(5)) * exp(-.5 * sum(((X-[params(1) params(2)])*pinv([params(3) params(4); params(4) params(3)]'*[params(3) params(4); params(4) params(3)])).*(X-[params(1) params(2)]),2));

% bounds
lb = [min(xx(:)) min(yy(:)) 0 -1 mnI]; %#ok<*NASGU>
ub = [max(xx(:)) max(yy(:)) 10 1 mnI+1];
        
% least-squares
options = optimoptions('lsqcurvefit', 'Display', 'none');
        
try
    phat = lsqcurvefit(gfun, par0, X, I(:), lb, ub, options);
catch
    phat = par0;
end
        
%         try
%             phat = lsqcurvefit(gfun, phat, X, I(:), lb, ub, options);
%         end
        
%         [phat,R,~,COVB] = nlinfit(X, I(:), gfun, par0);
%         CI = nlparci(phat, R, 'covar', COVB);
        
% convert paramters
mu = phat(1:2);
C = [phat(3) phat(4); phat(4) phat(3)]'*[phat(3) phat(4); phat(4) phat(3)];
        
        
stat = struct();
stat.gfun = gfun;
stat.phat = phat;

% get r2
Ihat = gfun(phat, X);
r2 = rsquared(I(:), Ihat(:));
stat.r2 = r2;

% convert multivariate gaussian to ellipse
trm1 = (C(1) + C(4))/2;
trm2 = sqrt( ((C(1) - C(4))/2)^2 + C(2)^2);

% half widths
l1 =  trm1 + trm2;
l2 = trm1 - trm2;

% convert to sqrt of area to match Rosa et al., 1997
stat.mu = mu;
stat.C = C;
stat.ar = sqrt(2 * l1 * l2);