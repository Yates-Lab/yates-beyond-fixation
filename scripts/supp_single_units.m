
datadir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'hires');

flist = dir(fullfile(datadir, '*.mat'));

S = struct();

for ex = 1:numel(flist)

    Exp = load(fullfile(datadir, flist(ex).name));

    sessname = strrep(flist(ex).name, '.mat', '');
    cids = arrayfun(@(x) x, Exp.osp.cids(Exp.osp.isiV < .1), 'uni', 0);
    S.(sessname) = struct('cids', Exp.osp.cids, 'isiV', Exp.osp.isiV, 'uQ', Exp.osp.uQ, 'cgs', Exp.osp.cgs, 'cgs2', Exp.osp.cgs2);
    fprintf('%s\t', sessname)
    fprintf('%d, ', cids{:})
    fprintf('\n')
end

%% save and copy to server
outputdir = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), 'stim_movies');
fname = fullfile(outputdir, 'unitstats.mat');
save(fname, '-v7.3', '-struct', 'S')

%%
server_string = 'jake@bancanus'; %'jcbyts@sigurros';
serverdir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/stim_movies/'; %/home/jcbyts/Data/MitchellV1FreeViewing/stim_movies/';

command = 'scp ';
command = [command fname ' '];
command = [command server_string ':' serverdir];

system(command)

fprintf('%s\n', fname)

%%

% arrayfun(@(x) x.cid, W)
% arrayfun(@(x) x.depth, W)
% arrayfun(@(x) x.isiRate, W)


cc = cc + 1;
W(cc)
figure(1); clf
plot(W(cc).lags, W(cc).isi);