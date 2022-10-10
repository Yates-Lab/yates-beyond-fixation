function h5addSpikesorting(Exp, fname, spike_sorting)
% h5addSpikeSorting(Exp, fname, spike_sorting)

% add spike sorting
goodspikes = ismember(Exp.osp.clu, Exp.osp.cids);

try
    field = 'cids';
    h5create(fname, ['/Neurons/' spike_sorting '/' field], size(Exp.osp.cids))
    h5write(fname, ['/Neurons/' spike_sorting '/' field], Exp.osp.cids)
    fprintf('Successfully added [%s]\n', field)
catch
    fprintf('Failed to add [%s]\n', field)
end

try
    field = 'times';
    h5create(fname, ['/Neurons/' spike_sorting '/' field], size(Exp.osp.st(goodspikes)))
    h5write(fname, ['/Neurons/' spike_sorting '/' field], Exp.osp.st(goodspikes))
    fprintf('Successfully added [%s]\n', field)
catch
    fprintf('Failed to add [%s]\n', field)
end


try
    field = 'cluster';
    h5create(fname, ['/Neurons/' spike_sorting '/' field], size(Exp.osp.clu(goodspikes)))
    h5write(fname, ['/Neurons/' spike_sorting '/' field], Exp.osp.clu(goodspikes))
    fprintf('Successfully added [%s]\n', field)
catch
    fprintf('Failed to add [%s]\n', field)
end

try
    field = 'cgs';
    h5create(fname, ['/Neurons/' spike_sorting '/' field], size(Exp.osp.cgs))
    h5write(fname, ['/Neurons/' spike_sorting '/' field], Exp.osp.cgs)
    fprintf('Successfully added [%s]\n', field)
catch
    fprintf('Failed to add [%s]\n', field)
end    

try
    field = 'peakval';
    h5create(fname, ['/Neurons/' spike_sorting '/' field], size([W.peakval]))
    h5write(fname, ['/Neurons/' spike_sorting '/' field], [W.peakval])
    fprintf('Successfully added [%s]\n', field)
catch
    fprintf('Failed to add [%s]\n', field)
end    

try
    field = 'troughval';
    h5create(fname, ['/Neurons/' spike_sorting '/' field], size([W.troughval]))
    h5write(fname, ['/Neurons/' spike_sorting '/' field], [W.troughval])
    fprintf('Successfully added [%s]\n', field)
catch
    fprintf('Failed to add [%s]\n', field)
end

try
    field = 'ciratio';
    exci = reshape([W.ExtremityCiRatio], 2, []);
    h5create(fname, ['/Neurons/' spike_sorting '/' field], size(exci))
    h5write(fname, ['/Neurons/' spike_sorting '/' field], exci)
    fprintf('Successfully added [%s]\n', field)
catch
    fprintf('Failed to add [%s]\n', field)
end

fprintf('Done\n')