%%

fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'FixFlashGabor', 'Gabor', 'BackImage'}, 'GazeContingent', true, 'includeProbe', true);

%% no gaze correction, no fixation point
fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'FixFlashGabor'}, 'GazeContingent', false, 'includeProbe', false);

%% gaze correction, no fixation point
fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'FixFlashGabor'}, 'GazeContingent', true, 'includeProbe', false);
%%
fname = make_stimulus_file_for_py(Exp, S, 'stimlist', {'FixFlashGabor'}, 'GazeContingent', false, 'includeProbe', true);