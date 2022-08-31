function r2 = rsquared(rtrue, rhat, flatten, rbar)
% r2 = rsquared(rtrue, rhat, flatten, rbar)

if ~exist('flatten', 'var')
    flatten = true;
end

if ~exist('rbar', 'var')
    if flatten
        rbar = nanmean(rtrue(:));
    else
        rbar = nanmean(rtrue);
    end
end

if flatten
    r2 = 1-(nansum((rtrue(:)-rhat(:)).^2))/(nansum((rtrue(:)-rbar).^2));
else
    r2 = 1-(nansum((rtrue-rhat).^2))./(nansum((rtrue-rbar).^2));
end
    