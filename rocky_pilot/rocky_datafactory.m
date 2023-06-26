function Exp = rocky_datafactory(exnum)


switch exnum

    case 1
        fname = '~/Downloads/r20230608.mat/';

    case 2
        fname = '~/Downloads/r20230608_dpi_redo.mat/';
        
    case 3
        
        fname = '~/Downloads/r20230618_dpi2.mat/';

end


fprintf('Loading [%s]\n', fname)
evalc('Exp = load(fname);');
fprintf('Done\n')