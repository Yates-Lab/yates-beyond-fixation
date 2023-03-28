
%% Set paths

% to hack my code, you need to set the SERVER Directory
setpref('FREEVIEWING', 'SERVER_DATA_DIR', '~/Dropbox/MarmoLabWebsite/PSA/DDPI_Before_Processed/')
setpref('FREEVIEWING', 'PROCESSED_DATA_DIR', '~/Dropbox/MarmoLabWebsite/PSA/DDPI_Before_Processed/')

S = struct();

S.rawFilePath = 'Allen_2022-06-10_13-05-49_V1_64b';
S.processedFileName = 'allen_20220610';
S.spikeSorting = 'kilo';

%% Run import

fname = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), S.rawFilePath, [S.processedFileName '.mat']);

if exist(fname, 'file')
    Exp = load(fname);
else
    Exp = io.importFreeViewing(S);
    save(fname, '-v7.3', '-struct', 'Exp')
end
