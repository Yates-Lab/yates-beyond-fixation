
%% Set paths

% to hack my code, you need to set the SERVER Directory
setpref('FREEVIEWING', 'SERVER_DATA_DIR', '~/Dropbox/MarmoLabWebsite/PSA/DDPI_Before_Processed/')
setpref('FREEVIEWING', 'PROCESSED_DATA_DIR', '~/Dropbox/MarmoLabWebsite/PSA/DDPI_Before_Processed/')

S = struct();

S.rawFilePath = 'Allen_2022-08-05_13-24-54_V1_64b';
S.processedFileName = 'allen_20220805';
S.spikeSorting = 'kilo';

%% Run import

fname = fullfile(getpref('FREEVIEWING', 'PROCESSED_DATA_DIR'), S.rawFilePath, [S.processedFileName '.mat']);

if exist(fname, 'file')
    Exp = load(fname);
else
    Exp = io.importFreeViewing(S);
    save(fname, '-v7.3', '-struct', 'Exp')
end
