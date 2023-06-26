function varargout = addFreeViewingPaths(user)
% set paths for FREEVIEWING projects
% this assumes you are running from the FREEVIEWING folder
% test git
if nargin < 1
    error('addFreeViewingPaths: requires a user argument')
end

switch user

    case 'jakelaptop'
        % we need marmoview in the path for the stimulus objects to
        % regenerate properly
        marmoViewPath = '~/Documents/MATLAB/MarmoV5/';
        % we only need marmopipe to import raw data
        marmoPipePath = '~/Documents/MATLAB/MarmoPipe/'; %'~/Dropbox/MarmoLabWebsite/PSA/Code/';
        % where the data live
        dataPath = '~/Dropbox/Datasets/Mitchell/';
        
        % processed data:
        setpref('FREEVIEWING', 'PROCESSED_DATA_DIR', dataPath)
        setpref('FREEVIEWING', 'SERVER_DATA_DIR', '/Volumes/mitchelllab/Data/')
        
    otherwise
        error('addFreeViewingPaths: I don''t know this user')
end

projectPath = fileparts(mfilename('fullpath'));
addpath(fullfile(projectPath, 'code'))

if ~isempty(marmoPipePath)
    addpath(marmoPipePath)
    addMarmoPipe
end

if ~isempty(marmoViewPath)
    addpath(marmoViewPath)
    addpath(fullfile(marmoViewPath, 'SupportFunctions'))
    addpath(fullfile(marmoViewPath, 'SupportData'))
    addpath(fullfile(marmoViewPath, 'Settings'))
end

if nargout == 1
    varargout{1} = dataPath;
end


    