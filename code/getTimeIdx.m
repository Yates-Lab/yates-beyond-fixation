function [idx, epoch] = getTimeIdx(times, epoch_starts, epoch_stops)
% idx = getEpochIdx(times, epoch_starts, epoch_stops)

numEpochs = numel(epoch_starts);
assert(numEpochs==numel(epoch_stops), 'getEpochIdx: epoch boundaries don''t match')

idx = false(numel(times), 1);
epoch = zeros(numel(times), 1);
for iEpoch = 1:numEpochs
    ix = times > epoch_starts(iEpoch) & times < epoch_stops(iEpoch);
    idx(ix) = true;
    epoch(ix) = iEpoch;
end

