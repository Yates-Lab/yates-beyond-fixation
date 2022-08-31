%% Setup folder where the data live

PROCESSED_DATA_DIR = '/Users/jcbyts/Dropbox/MarmoLabWebsite/PSA/MT_RF/Processed22';
OUTPUT_DIR = fullfile(PROCESSED_DATA_DIR, 'processed');
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR)
end

% run import on all
FileList = arrayfun(@(x) x.name, dir(fullfile(PROCESSED_DATA_DIR, '*.mat')), 'uni', 0);
numFiles = numel(FileList);
FilesToCopy = cell(numFiles,1);

addpath Analysis/MT_RF/

%%
% numFiles = 1;
% FilesToCopy = cell(numFiles,1);
%**************** then can batch script across files
for zkk = 1:numFiles

    try
    clear Exp;  % previous run
    close all;
    %******* Start new run
    FileTag = FileList{zkk};


    %% Grab Exp File that was previously imported
    ExpFile = fullfile(PROCESSED_DATA_DIR,FileTag);
    fprintf('Loading processed Exp file, %s\n',FileTag)
    load(ExpFile);  % struct Exp will be loaded
    disp('File loaded');

    %% New Motion Analaysis of Space, with Sac and Motion Selectivity

    SPClust = 1;  % need to revise to loop through all 64 units (and faster)
    GRID.box = [0,0,30,30];  % center x,y, width an d height
    GRID.div = 2.0;

    fname = sprintf('%s_%d_%d_%d_%d_%d.mat', strrep(FileTag, '.mat', ''), GRID.box(1), GRID.box(2), GRID.box(3), GRID.box(4), GRID.div);

    fout = fullfile(OUTPUT_DIR, fname);
    if exist(fout, 'file')==2
        load(fout)
    else
        % [MoStimX,MoSacX,MoStimY] = Forage.StimMatrix_ForageMotionSpatialSacKernel(Exp,GRID);  %step 1
        [MoStimX,MoSacX,MoStimY,MoStimPX] = Forage.StimMatrix_ForageMotionSpatialSacKernel(Exp,[],GRID);  %step 1
        % new addition is MoStimPX -- it is a vector that gives dot speed,
        % where motion direction (1 of 16) was in StimX ... bit space hungry,
        % so unfortunate, but need speed for each dot

        % in the original all dots had 15 deg/sec as the speed
        % probably not worth analyzing speed for Nat Comm paper ... nearly all
        % cells prefer higher speeds up to 16 or beyond, if anything could just
        % do your vector analysis and use different length vectors
        save(fout, '-v7.3', 'MoStimX', 'MoSacX', 'MoStimY', 'MoStimPX', 'GRID')
    end

    FilesToCopy{zkk} = fout;
    end

end


%% copy to server
% server_string = 'jcbyts@sigurros';
% output_dir = '/home/jcbyts/Data/MitchellV1FreeViewing/grating_subspace';


server_string = 'jake@bancanus';
output_dir = '/home/jake/Data/Datasets/MitchellV1FreeViewing/MT_RF';

data_dir = OUTPUT_DIR;

command = 'scp ';
command = [command fullfile(data_dir, '*.mat') ' '];
command = [command server_string ':' output_dir];

system(command)

disp('Done')


   

