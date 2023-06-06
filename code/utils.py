"""Local utility functions."""

###################################################################################################
###################################################################################################
import numpy as np
from scipy.stats import spearmanr

from convnwb.io import load_nwbfile

from spiketools.utils.data import make_row_orientation
from spiketools.utils.epoch import epoch_data_by_range, epoch_spikes_by_range
from spiketools.utils.base import count_elements
from spiketools.utils.trials import recombine_trial_data
from spiketools.utils.extract import get_range, get_values_by_time_range, get_values_by_times



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