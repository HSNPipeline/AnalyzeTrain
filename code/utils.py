"""Local utility functions."""
import numpy as np
from scipy.stats import spearmanr, ttest_ind
from convnwb.io import load_nwbfile

#from spiketools.utils.data import make_row_orientation
from spiketools.utils.epoch import epoch_spikes_by_range,epoch_data_by_time, epoch_data_by_range
from spiketools.utils.extract import (get_range, get_values_by_times,
                                      get_value_by_time,get_inds_by_times,get_values_by_time_range)
from spiketools.utils.base import count_elements
from spiketools.utils.trials import recombine_trial_data
from spiketools.utils.checks import check_axis
from spiketools.spatial.distance import compute_distance, get_closest_position

###################################################################################################
###################################################################################################
def group_array_by_key(keys, values):
    result = {}
    for key, value in zip(keys, values):
        if key not in result:
            result[key] = []
        result[key].append(value)
    return result

def select_movement(data, move_starts, move_stops, recombine=True):
    """Helper function to select data from during navigation periods."""

    times_trials, values_trials = epoch_data_by_range(\
        data.timestamps[:], data.data[:], move_starts, move_stops)

    if not recombine:
        return times_trials, values_trials
    else:
        times, values = recombine_trial_data(times_trials, values_trials)
        return times, values
    

def get_values_by_times(timestamps, values, timepoints, time_threshold=None,
                        drop_null=True, axis=None):
    axis = check_axis(axis, values)
    inds = get_inds_by_times(timestamps, timepoints, time_threshold, drop_null)

    if drop_null:
        outputs = values.take(indices=inds, axis=check_axis(axis, values))
    else:
        outputs = np.full([np.atleast_2d(values).shape[0], len(timepoints)], np.nan)
        mask = inds >= 0
        outputs[:, np.where(mask)[0]] = values.take(indices=inds[mask], axis=axis)
        outputs = np.squeeze(outputs, axis=0 if outputs.shape == (1, 1) else None)

    return outputs,inds


def drop_nan_from_array(sub_arr):
    return sub_arr[~np.isnan(sub_arr)]


def get_trial_pos_times(position, timestamps, starts, stops, targets):
    """Extract trial-organized spike times around position(s) of interest."""
    
    pos_times = np.zeros(len(starts))
    for ind, (cstart, cstop, ctarget) in enumerate(zip(starts, stops, targets)):
        t_times, t_pos = get_values_by_time_range(timestamps, position, cstart, cstop)
        p_ind = get_closest_position(t_pos, ctarget)
        pos_times[ind] = t_times[p_ind]
        
    return pos_times