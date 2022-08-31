function qf = qfsmooth3D(dims, lambda)
% qf = qfsmooth3D(dims, lambda)
    
    if nargin < 2
        lambda = 1;
    end
    
    numt = dims(1);
    numx = dims(2);
    numy = dims(3);
    
    if numel(lambda)==2
        lambda_space = lambda(2);
        lambda_time = lambda(1);
    else
        lambda_space = lambda;
        lambda_time = lambda;
    end
        
    Dt = lambda_time*qfsmooth1D(numt);
    Dx = lambda_space*qfsmooth1D(numx);
    Dy = lambda_space*qfsmooth1D(numy);

    
    
    It = speye(numt);
    Ix = speye(numx);
    Iy = speye(numy);
    

%     qf = kron(Dt, kron(Ix, Iy)) + kron(It, kron(Dx, Iy)) + kron(It, kron(Ix, Dy));
    qf = kron(kron(Ix, Iy), Dt) + kron(kron(Dx, Iy),It) + kron(kron(Ix, Dy),It);
end