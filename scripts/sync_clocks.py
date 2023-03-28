import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.stats import linregress
from scipy.interpolate import interp1d

def convertSamplesToTimestamps(t, y):
    # Assumes signal is roughly a bistable oscillating digital signal. Picks threshold between max and min.
    thresh = (y.max() + y.min()) / 2
    # +1 returns the first timestamp where signal is high
    pulse_idx = np.nonzero(np.diff((y > thresh).astype(np.int8)) == 1)[0] + 1 
    return t[pulse_idx] 


def convertTimestampsToIntervals(t):
    intervals = np.diff(t)

    # Pulses come in ~.5 second width followed by 1.5, 2, ..., 4.5s widths. 
    # First we determine alginment (i.e. if we started sampling in the .5s interval)

    first_interval = intervals[0]
    second_interval = intervals[1]

    fix_start = 0
    var_start = 1
    if first_interval > second_interval:
        fix_start = 1
        var_start = 0

    fixed_intervals = intervals[fix_start::2]
    variable_intervals = intervals[var_start::2]
    
    # Assert that the fixed interval is less than the minimum variable interval
    assert fixed_intervals.max() < variable_intervals.min()

    fixed_duration = fixed_intervals.mean()
    data = np.round(intervals / fixed_duration)
    
    return data

def alignTimestampToTimestamp(t1, t2, debug=False):
    """
        Calculates alignment function that linearly transforms t1 into t2.

        Parameters:
            t1 (np.array): [# timestamps] Vector containing timestamps recorded on one device
            t2 (np.array): [# timestamps] Vector containing timestamps recorded on a second device
        Returns:
            slope (float): slope of transfer function (to align t1 to t2 calculate t1*slope + intercept)
            intercept (float): intercept of transfer function (to align t1 to t2 calculate t1*slope + intercept)

    """

    d1 = convertTimestampsToIntervals(t1)
    d2 = convertTimestampsToIntervals(t2)

    if debug:
        print(f'Data 1 (length - {len(d1)}):')
        print(d1[:10])
        print(f'Data 2 (length - {len(d2)}):')
        print(d2[:10])
    
    # 1 in 1,000,000 odds of errant 100%
    n_elements = len(np.unique(d1)) - 1
    min_length = int(np.ceil(np.log(1000000) / np.log(n_elements)))
    if debug:
        print(f'min_length: {min_length}')
    # calculate coherence for every lag of d1, including positive and over extending
    d1_idx = np.arange(len(d1))
    d2_idx = np.arange(len(d2))
    lags = np.arange(1-len(d1)+min_length, len(d2)-min_length)
    coherence = np.zeros(len(lags), dtype=np.int32)
    len_frames = np.zeros(len(lags), dtype=np.int32)
    for iL in range(len(lags)):
        lag = lags[iL]
        d1_idx_lagged = d1_idx + lag
        start_idx = np.max([np.min(d1_idx_lagged), np.min(d2_idx)])
        end_idx = np.min([np.max(d1_idx_lagged), np.max(d2_idx)])
        d1_mask = np.logical_and(start_idx<=d1_idx_lagged,d1_idx_lagged<=end_idx)
        d2_mask = np.logical_and(start_idx<=d2_idx,d2_idx<=end_idx)
        # print(f'Lag {lag}\n\td1_mask {d1_mask.astype(np.int8)}\n\td2_mask {d2_mask.astype(np.int8)}')
        d1_frame = d1[d1_mask]
        d2_frame = d2[d2_mask]
        coherence[iL] = np.sum(d1_frame == d2_frame)
        len_frames[iL] = np.sum(d1_mask)
    
    coherence_pct = coherence / len_frames
    sync_lag = lags[np.argmax(coherence_pct)]
    
    if coherence_pct.max() < 1:
        logging.warn(f'(alignTimestampToTimestamp) Could not find interval with 100% Coherence. Best match {coherence_pct}.')

    if np.sum(coherence_pct == 1) > 1:
        logging.warn(f'(alignTimestampToTimestamp) Multiple fully coherent segments found. Result may be misaligned.')

    if debug:
        nz_mask = coherence != 0
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(lags[nz_mask], coherence[nz_mask])
        plt.plot(lags[np.argmax(coherence)], coherence[np.argmax(coherence)], 'rx')
        plt.xlabel('Lags')
        plt.ylabel('Coherence')
        plt.subplot(2,1,2)
        plt.plot(lags[nz_mask], coherence_pct[nz_mask])
        plt.plot(lags[np.argmax(coherence)], coherence_pct[np.argmax(coherence)], 'rx')
        plt.xlabel('Lags')
        plt.ylabel('% Coherence')
        plt.show()

    # Retrieve timestamps for all datapoints which were sampled by both inputs
    d1_idx_lagged = d1_idx + sync_lag
    start_idx = np.max([np.min(d1_idx_lagged), np.min(d2_idx)])
    end_idx = np.min([np.max(d1_idx_lagged), np.max(d2_idx)])
    d1_mask = np.logical_and(start_idx<=d1_idx_lagged,d1_idx_lagged<=end_idx)
    d2_mask = np.logical_and(start_idx<=d2_idx,d2_idx<=end_idx)
    
    t1_sync = t1[np.append(d1_mask,False)]
    t2_sync = t2[np.append(d2_mask,False)]

    # Perform linear regression to map t1 onto t2
    slope, intercept, r, p, se = linregress(t1_sync, t2_sync)

    if debug:
        error = t2_sync - (t1_sync* slope + intercept)

        # Plot Times & Errors
        plt.figure(figsize=(8,10))
        plt.subplot(211)
        plt.plot(t1_sync, t2_sync, 'ko', label='Pulse Times')
        plt.plot(t1_sync, t1_sync * slope + intercept, 'r-', label='Fitted Transform')
        plt.title(f'Regressed Fit - Slope: {slope:.02F}, Offset: {intercept:.02F}s')
        plt.xlabel('t2')
        plt.ylabel('t1')
        plt.legend()
        plt.subplot(212)
        plt.hist(error*1000, 20)
        plt.title('Residuals')
        plt.xlabel('t (ms)')
        plt.ylabel('Count')
        plt.show()
    
    return slope, intercept

def alignContToCont(a_t, a_sync, b_t, b_sync, debug=0):
    """
        Aligns clock of a to that of clock b using a shared sync line. 

        Parameters:
            a_t (np.array): [# samples in a] vector of timepoints for each sample in a
            a_sync (np.array): [# samples in a] vector of digital reads from the shared sync line
            b_t (np.array): [# samples in b] vector of timepoints for each sample in b
            b_sync (np.array): [# samples in b] vector of digital reads from the shared sync line
            debug (boolean): if set to True, plots alignment and residuals.

        Returns:
            a_t_aligned (np.array): [# samples in a] affine transform of a_t to match b_t given shared sync line
    """
    a_ts = convertSamplesToTimestamps(a_t, a_sync) # This gives us timestamps for each pulse

    b_ts = convertSamplesToTimestamps(b_t, b_sync)

    slope, intercept = alignTimestampToTimestamp(a_ts, b_ts, debug=debug)

    return a_t * slope + intercept


def test_alignContToCont(plot=False):

    # Generate a long binary signal of the type generated by the arduino
    # Take two snippit in the middle, resample one to a higher framerate and test that the offset and gain can be recovered
    # Test both ways
    np.random.seed(123) 

    src_dur = 1000
    src_fs = 3333
    src_amp = 5

    src_t = np.arange(src_dur*src_fs) / src_fs
    src_y = np.zeros(len(src_t))

    pulse_dur = .25
    set_int_dur = .5
    rand_int_durs = np.arange(1,5,.5)

    next_rand = True
    t_gen = 0
    while t_gen < src_dur:
        t_step = np.random.choice(rand_int_durs) if next_rand else set_int_dur
        t_gen += t_step
        next_rand = not next_rand
        src_mask = np.logical_and(t_gen <= src_t, src_t < t_gen + pulse_dur)
        src_y[src_mask] = src_amp

    src_interp = interp1d(src_t, src_y)

    if plot:
        plt.figure()
        plt.plot(src_t, src_y)
        plt.xlim([0,10])
        plt.show()

    def sampleSync(start, end, fs, slope):
        t = np.arange(start, end, 1/fs) # Samples in src time
        sync = src_interp(t) # Sample the src signal
        t = (t - t[0]) * slope # Skew the clock into sampled time
        return t, sync

    # Test 1 Aligned sequence totally contained in the second and reverse
    print('Running Test 1 - b contained in a')
    a_start = 4.4
    a_end   = 500
    a_fs    = 500
    a_slope = 1.1

    a_t, a_sync = sampleSync(a_start, a_end, a_fs, a_slope)

    b_start = 34.4
    b_end   = 130
    b_fs    = 1000
    b_slope = 1

    b_t, b_sync = sampleSync(b_start, b_end, b_fs, b_slope)

    a_ts = convertSamplesToTimestamps(a_t, a_sync)
    b_ts = convertSamplesToTimestamps(b_t, b_sync)

    slope, intercept = alignTimestampToTimestamp(a_ts, b_ts, debug=plot)

    true_slope = b_slope / a_slope
    print(f'\tTrue Slope {true_slope:.5f} | Estimated Slope {slope:.5f} | Difference {(true_slope - slope)/true_slope*100:.5f}%')

    true_intercept = a_start - b_start
    print(f'\tTrue Offset {true_intercept:.5f} | Estimated Offset {intercept:.5f} | Difference {(true_intercept - intercept)/true_intercept*100:.5f}%')

    if np.abs((true_slope - slope)/true_slope) < .00001 and np.abs(true_intercept - intercept) < .01:
        print('Test 1 Passed!')
    else:
        print('Test 1 Failed')

    # Test 2 
    print('Running Test 2 - b longer than a')
    a_start = 4.4
    a_end   = 500
    a_fs    = 500
    a_slope = 1.1

    a_t, a_sync = sampleSync(a_start, a_end, a_fs, a_slope)

    b_start = 204.4
    b_end   = 600
    b_fs    = 900
    b_slope = 1

    b_t, b_sync = sampleSync(b_start, b_end, b_fs, b_slope)

    a_ts = convertSamplesToTimestamps(a_t, a_sync)
    b_ts = convertSamplesToTimestamps(b_t, b_sync)

    slope, intercept = alignTimestampToTimestamp(a_ts, b_ts, debug=plot)

    true_slope = b_slope / a_slope
    print(f'\tTrue Slope {true_slope:.5f} | Estimated Slope {slope:.5f} | Difference {(true_slope - slope)/true_slope*100:.5f}%')

    true_intercept = a_start - b_start
    print(f'\tTrue Offset {true_intercept:.5f} | Estimated Offset {intercept:.5f} | Difference {(true_intercept - intercept)/true_intercept*100:.5f}%')

    if np.abs((true_slope - slope)/true_slope) < .00001 and np.abs(true_intercept - intercept) < .01:
        print('Test 2 Passed!')
    else:
        print('Test 2 Failed...')

if __name__ == "__main__":
    test_alignContToCont(plot=True)