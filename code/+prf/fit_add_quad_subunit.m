function NIMf = fit_add_quad_subunit(NIM0, Robs, stim, train_inds, test_inds, fname, varargin)
% NIMf = fit_add_quad_subunit(NIM0, Robs, stim, train_inds, test_inds, fname, ip)
% Inputs:
%   NIM0 (NIMclass) - an already fit NIM or sNIM object
%   Robs (T x 1)    - binned spike counts
%   stim (cell)     - each element of the cell-array is a (T x m) stimulus
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
%   NIMf - fitted NIM model - a NIM or sNIM object

ip = inputParser();
ip.addParameter('spkNL', 'softplus')
ip.addParameter('xtargets', 1)
ip.addParameter('rank', 1)
ip.addParameter('debug', 0)
ip.addParameter('overwrite', 0)
ip.addParameter('verbose', 0)
ip.addParameter('reg_path', 0)
ip.addParameter('nosave', 0)
ip.parse(varargin{:})

[pathto, fn, ~] = fileparts(fname);
fn = ['nim_' fn '.mat'];
fname = fullfile(pathto, fn);

if exist(fname, 'file') && ~ip.Results.overwrite
    load(fname)
    return
end
    

% NIM: linear plus suppressive; like Butts et al (2011)
% Add an inhibitory input (with delayed copy of GLM filter and fit 
% delayed_filt = NIM.shift_mat_zpad( NIM0.subunits(1).filtK, 4 );
% NIM1 = NIM0.add_subunits( {'rectlin'}, -1, 'init_filts', {delayed_filt} );
NIM1 = NIM0.add_subunits({'quad'}, 1);
NIM1 = NIM1.fit_filters( Robs, stim, train_inds );
% Allow threshold of suppressive term to vary
NIM2 = NIM1.fit_filters( Robs, stim, 'fit_offsets', 1 ); % doesnt make huge difference
% Search for optimal regularization

if ip.Results.debug
    % Compare subunit filters (note delay of inhibition)
    NIM2.display_subunit_filters()
end

if ip.Results.reg_path   
    NIMf = NIM2.reg_path( Robs, stim, train_inds, test_inds); %, 'lambdaID', 'd2t' );
    NIMf = NIMf.fit_spkNL( Robs, stim, train_inds, 'silent', 1 );
else
    NIMf = NIM2;
end

if ~ip.Results.nosave
    save(fname, '-v7.3', 'NIMf')
end