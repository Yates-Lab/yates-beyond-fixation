function [O, OG, Xstims] = fit_add_offset_gain(NIM0, Robs, stim, opts, train_inds, test_inds, fname, varargin)
% Fit perisaccadic offset and gain terms on top of already fit NIM model
% [O, OG, Xstims] = fit_add_offset_gain(NIM0, Robs, stim, opts, train_inds, test_inds, fname, varargin)
% Inputs:
%   

ip = inputParser();
ip.addParameter('verbose', 0)
ip.addParameter('reg_path', 0)
ip.addParameter('overwrite', 0)
ip.addParameter('nosave', 0)
ip.parse(varargin{:});

if nargin < 5 || isempty(train_inds)
    NT = numel(Robs);
    test_inds = ceil(NT*2/5):ceil(NT*3/5);
    train_inds = setdiff(1:NT,test_inds);
end

[pathto, fn, ~] = fileparts(fname);
fn = ['og_' fn '.mat'];
fname = fullfile(pathto, fn);

if exist(fname, 'file') && ~ip.Results.overwrite
    load(fname)
    return
end

nkt = opts.num_lags_sac_pre*2;

% evaluate model to get the input to the OG model
[~, ~, modint0] = NIM0.eval_model(Robs, stim);
xproj = modint0.G;

Xsac = fliplr(makeStimRows(stim{2}, nkt));

% build parameters for new saccade basis stimulus
sacparams = repmat(struct('dims', [1 1 1], ... % first stim is 1 x 1 x 1 - it is the g(t) from the previous model
    'dt', 1e3/opts.fs_spikes, 'dx', 1, ...
    'up_fac', 1, ...
	'tent_spacing', [], ...
    'boundary_conds', [0 0 0], 'split_pts', []), 3, 1);

sacparams(2).dims = [1 nkt 1]; % second stimulus input is the sacade basis multipled by the 
sacparams(3).dims = [1 nkt 1]; % gain (saccade basis multipled by g(t))

O = NIM(sacparams, {'lin', 'lin'}, [1 1], 'spkNL', 'softplus', 'xtargets', [1, 2]);
O.spkNL = NIM0.spkNL; % use existing spkNL

Xstims ={xproj, Xsac, Xsac.*xproj};
O = O.fit_filters(Robs, Xstims, train_inds, 'silent', ~ip.Results.verbose);

% --- add gain term
OG = O.add_subunits( {'lin'}, 1, 'xtargs', 3);
OG = OG.fit_filters(Robs, Xstims, train_inds, 'silent', ~ip.Results.verbose);

% --- refit with regularization
O = O.set_reg_params( 'd2x', [0 100 100]'); % spatial smoothing
O = O.set_reg_params( 'l1', [0 1 1]');    % sparseness

O = O.fit_filters(Robs, Xstims, train_inds, 'silent', ~ip.Results.verbose);

% OG = OG.set_reg_params( 'd2t', [0 0 0]');   % temporal smoothing
OG = OG.set_reg_params( 'd2x', [0 100 100]'); % spatial smoothing
OG = OG.set_reg_params( 'l1', [0 1 1]');    % sparseness

OG = OG.fit_filters(Robs, Xstims, train_inds, 'silent', ~ip.Results.verbose);

% figure(2); clf
% subplot(1,2,1)
% plot(OG.subunits(2).filtK); hold on
% plot(O.subunits(2).filtK);
% 
% title('offset')
% 
% subplot(1,2,2)
% plot(OG.subunits(3).filtK)
% title('gain')
%%

% --- learn hyperparameters for regularization
if ip.Results.reg_path
    O = O.reg_path(Robs, Xstims, train_inds, test_inds, 'silent', ~ip.Results.verbose);
    OG = OG.reg_path(Robs, Xstims, train_inds, test_inds, 'silent', ~ip.Results.verbose);
end

if ~ip.Results.nosave
    save(fname, '-v7.3', 'O', 'OG', 'Xstims')
end
