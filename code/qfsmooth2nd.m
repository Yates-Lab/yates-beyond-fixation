function [qf,D,qfx,qfy] = qfsmooth2nd(numx,numy)
    %[qf] = qfsmooth(numx,numy)
    %Create a quadratic form for smoothness regularization, with D a
    %second partial derivative operator and qf = D'*D
    D = zeros((numx-1)*numy + (numy-1)*numx,numx*numy);
    for jj = 1:numy
        for ii = 1:numx-1
            [xi, yi] = meshgrid(1:numx,1:numy);
            dd = (xi == ii & yi == jj) - (xi == ii + 1 & yi == jj);
            dd = dd + ((xi == ii & yi == jj) - (xi == ii - 1 & yi == jj));
            
            D(ii + (jj-1)*(numx-1),:) = dd(:);
        end
    end
    
    for jj = 1:numy-1
        for ii = 1:numx
            [xi, yi] = meshgrid(1:numx,1:numy);
            dd = (xi == ii & yi == jj) - (xi == ii & yi == jj + 1);
            dd = dd + (xi == ii & yi == jj) - (xi == ii & yi == jj - 1);
            
            D((numx -1)*numy + ii + (jj-1)*numx,:) = dd(:);
        end
    end
    
    qf  = D'*D;
    D1  = D(1:round(end/2),:);
    qfx = D1'*D1;
    D1  = D(round(end/2)+1:end,:);
    qfy = D1'*D1;
end