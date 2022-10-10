function LNf = fit_LN(Robs, stim, params, train_inds, test_inds, fname, varargin)
% Fit LNP model using NIMclass
% LNf = fit_LN(Robs, stim, params, train_inds, test_inds, fname, varargin)
% Inputs:
%   Robs (T x 1)    - binned spike counts
%   stim (cell)     - each element of the cell-array is a (T x m) stimulus
%   params (struct) - struct array for NIM parameters
%   train_inds      - indices to train with
%   test_inds       - indices to test
%   fname           - file (load if exists)
% Optional (as argument pairs): these all specify how NIM is called
%   'spkNL'     - spike nonlinearity {'softplus', 'exp'}
%   'xtargets'  - which stim do the filters target (default: 1)
%   'rank'      - rank of the filters (if is an integer, use sNIM)
%   'debug'     - print debugging output (default: false)
%   'overwrite' - overwrite file (default: false)
%   'verbose'   - print model fitting output (default: false)
%   'reg_path'  - learn hyperparameters with cross-validation (default: false)
% Output:
%   LNf - fitted LN model - a NIM or sNIM object

ip = inputParser();
ip.addParameter('spkNL', 'softplus')
ip.addParameter('xtargets', 1)
ip.addParameter('rank', 1)
ip.addParameter('debug', 0)
ip.addParameter('overwrite', 0)
ip.addParameter('verbose', 0)
ip.addParameter('reg_path', 1)
ip.addParameter('nosave', 0)
ip.parse(varargin{:})

% check if the file exists and run if it does
[pathto, fn, ~] = fileparts(fname);
fn = ['ln_' fn '.mat'];
fname = fullfile(pathto, fn);

if exist(fname, 'file') && ~ip.Results.overwrite
    load(fname)
    return % exit if a previous fit was successfully loaded
end

% initial model
LN0  = NIM( params, {'lin'}, 1, 'xtargets', ip.Results.xtargets, 'spkNL', ip.Results.spkNL);

if isnumeric(ip.Results.rank)
    % space-time seperable version
    sLN    = sNIM(LN0, ip.Results.rank, params);
else
    sLN    = LN0; % use full NIM
end

% inital maximum-likelihood fit
sLN    = sLN.fit_filters(Robs, stim, train_inds, 'silent', ~ip.Results.verbose );

% add regularization
LN0 = sLN;
if any(strfind(fname, 'Fs_1000'))
    LN0 = LN0.set_reg_params( 'd2t', [100 100]);   % temporal smoothing
    LN0 = LN0.set_reg_params( 'd2x', [40 40] ); % spatial smoothing
    LN0 = LN0.set_reg_params( 'l1', [2 2] );    % sparseness
else
    LN0 = LN0.set_reg_params( 'd2t', [1 1]);   % temporal smoothing
    LN0 = LN0.set_reg_params( 'd2x', [40 40] ); % spatial smoothing
    LN0 = LN0.set_reg_params( 'l1', [2 1] );    % sparseness
end

% fit with regularization
sLN    = sNIM(LN0, 1, params(1));
sLN    = sLN.fit_filters(Robs, stim, train_inds, 'silent', ~ip.Results.verbose );
LN1    = sLN;

if ip.Results.reg_path
    % Optimize all models
    LNf  = LN1.reg_path( Robs, stim, train_inds, test_inds , 'silent', ~ip.Results.verbose);
    LNf  = LNf.fit_spkNL( Robs, stim, train_inds, 'silent', ~ip.Results.verbose);
    
    if any(isnan(LNf.subunits(1).kt))
        LNf  = LN1.fit_spkNL( Robs, stim, train_inds, 'silent', ~ip.Results.verbose);
    end
        
else
    LNf  = LN1;
end

% saving
if ~ip.Results.nosave
    save(fname, '-v7.3', 'LNf')
end