function fname = h5_add_struct(fname, S, fpath)
% fname = h5_add_struct(fname, struct, fpath)
% Inputs:
%   fname: filename of h5 file
%   S: struct you want to add
%   fpath: where you want to add it 
%
% This does no error checking and will write to file so don't mess up!

fields = fieldnames(S);
for ifield = 1:numel(fields)
    try
        tmp = S.(fields{ifield});
        if iscellstr(S.(fields{ifield}))
            tmp = strjoin(S.(fields{ifield}), ','); % comma seperated list
            h5writeatt(fname, fpath, fields{ifield}, tmp);
        else
            h5create(fname, fullfile(fpath, fields{ifield}), size(S.(fields{ifield})));
            h5write(fname, fullfile(fpath, fields{ifield}), S.(fields{ifield}));
        end
        
    catch
        fprintf('h5addstruct: could not add [%s]\n', fields{ifield})

    end
end
