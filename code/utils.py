"""Local utility functions."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
from scipy import signal
from scipy.ndimage import label
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

def compute_t_occupancy(trial_bin, pos_bin, edges_trial, edges_pos, epochSize):
    trial_occupancy = np.zeros((len(edges_trial)-1, len(edges_pos)-1))
    for t, p in zip(trial_bin, pos_bin):
        if t > 0 and t <= len(edges_trial)-1 and p > 0 and p <= len(edges_pos)-1:
            trial_occupancy[t-1, p-1] += epochSize
    return trial_occupancy


def compute_firing_rates(units_fr, pos_bin, occ, g, numBins, epochSize):
    """
    Calculate place bins with smoothed firing rates
    
    Parameters:
    -----------
    units_fr : ndarray
        Firing rate matrix
    pos_bin : ndarray
        Position bin assignments
    occ : ndarray
        Occupancy time in each bin
    g : ndarray
        Gaussian smoothing kernel
    numBins : int
        Number of position bins
    epochSize : float
        Size of each time epoch in seconds
        
    Returns:
    --------
    place_bins : ndarray
        Smoothed firing rates across position bins
    """
    epoch_spikes = units_fr * epochSize
    
    # Count spikes in each position bin
    df = pd.DataFrame({
        'pos_bin': pos_bin,
        'fr_value': epoch_spikes
    })
    spike_count_by_position = df.groupby('pos_bin')['fr_value'].sum()
    spike_count_by_position_array = spike_count_by_position.values
    smoothed_spike_count = signal.convolve(spike_count_by_position_array, g, mode='same')
    
    if len(smoothed_spike_count) != numBins:
        empty_bins = occ == 0
        occ = occ[~empty_bins]
        smoothed_spike_count = smoothed_spike_count[:-1]
        
    smoothed_occ = signal.convolve(occ, g, mode='same')
    place_bins = smoothed_spike_count/smoothed_occ
    place_bins[smoothed_occ < 0.1] = np.nan
    
    return place_bins

def compute_trial_firing_rates(trial_bin, pos_bin, units_fr, edges_trial, edges_pos, trial_occupancy, kernelSize, epochSize):
    # Calculate spike counts per trial and position bin
    trial_spikes = np.zeros((len(edges_trial)-1, len(edges_pos)-1))
    for t, p, f in zip(trial_bin, pos_bin, units_fr):
        if t > 0 and t <= len(edges_trial)-1 and p > 0 and p <= len(edges_pos)-1:
            trial_spikes[t-1, p-1] += f * epochSize

    # Create and normalize gaussian window
    g = signal.windows.gaussian(kernelSize, std=1)
    g = g/np.sum(g) # normalize

    # Apply smoothing across trials
    smoothed_trials = np.full(trial_spikes.shape, np.nan) # initialize the smooth spikes per bin per trial
    smoothed_time_trials = np.full(trial_occupancy.shape, np.nan) # initialize the time per bin per trial
    for bin in range(trial_spikes.shape[0]):
        smoothed_trials[bin,:] = signal.filtfilt(g, 1, trial_spikes[bin,:])
        smoothed_time_trials[bin,:] = signal.filtfilt(g, 1, trial_occupancy[bin,:])

    # Calculate firing rates and mask low occupancy bins
    trial_place_bins = smoothed_trials/smoothed_time_trials # get the "pure firing rate"
    trial_place_bins[smoothed_time_trials < 0.1] = np.nan

    trial_fr = trial_spikes/trial_occupancy
    
    return trial_place_bins, trial_fr

def circular_shuffle_unit_fr(units_fr, n_shuffles=1000):
    """
    Perform circular shuffling on event firing rate data
    
    Parameters
    ----------
    event_fr : array-like
        Array of firing rates for events/trials
    n_shuffles : int, optional
        Number of shuffles to perform (default=100)
        
    Returns
    -------
    shuffled : ndarray
        Array of shuffled firing rates with shape (n_shuffles, len(event_fr))
    """
    units_fr = np.array(units_fr)
    shuffled = np.zeros((n_shuffles, len(units_fr)))
    
    for i in range(n_shuffles):
        # Generate random shift amount
        shift = np.random.randint(0, len(units_fr))
        # Perform circular shift
        shuffled[i] = np.roll(units_fr, shift)
        
    return shuffled

def get_significant_percentage(stats_vals,p_vals,increment,threshold = 0.05):

    ## Calculate the number of steps
    # Remove NaN values
    valid_mask = ~np.isnan(stats_vals)
    stats_vals = stats_vals[valid_mask]
    p_vals = p_vals[valid_mask]
    
    max_value = int(np.max(stats_vals))+1
    n_step = int(max_value/increment)
    categories = []
    significant_percentages = []
    for i in range(n_step):
        lower_bound = i*increment
        upper_bound = (i+1)*increment

        # Find indices and compute percentage of significant p values
        indx_in_category = np.where((stats_vals >=lower_bound)& (stats_vals < upper_bound))[0]
        if len(indx_in_category) > 0:
            p_vals_in_category = p_vals[indx_in_category]
            significant_p_vals = p_vals_in_category[p_vals_in_category<threshold]

            n_significant_p_vals = len(significant_p_vals)
            significant_percentage = (n_significant_p_vals / len(p_vals_in_category)) * 100
            category = upper_bound
        else:
            significant_percentage = None
            category = None
        
        significant_percentages.append(significant_percentage)
        categories.append(category)
    return categories,significant_percentages


def get_agreement_percentage(stats, p_values1, p_values2, increment=0.2,threshold = 0.05):
    """
    Calculate the percentage of agreement between two sets of p-values.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(stats)
    stats = stats[valid_mask]
    p_values1 = p_values1[valid_mask]
    p_values2 = p_values2[valid_mask]

    if len(stats) == 0:
        return [], [], [], []

    max_value = int(np.max(stats)) + 1
    num_steps = int(max_value / increment)
    categories = []
    agreement_percentages = []
    significant_percentages = []
    not_significant_percentages  = []

    for i in range(num_steps):  # Loop over increments
        lower_bound = i * increment
        upper_bound = (i + 1) * increment
        categories.append(upper_bound)
        indices_in_category = np.where((stats >= lower_bound) & (stats < upper_bound))[0]

        if len(indices_in_category) > 0:
            p_values1_in_category = p_values1[indices_in_category]
            p_values2_in_category = p_values2[indices_in_category]

            significant_indices = (p_values1_in_category < threshold) & (p_values2_in_category < threshold)
            not_significant_indices = (p_values1_in_category > threshold) & (p_values2_in_category > threshold)

            n_agree = np.sum(significant_indices)+np.sum(not_significant_indices)  # Count significant values

            # Calculate percentage
            agreement_percentages.append(n_agree / len(indices_in_category) * 100)
            significant_percentages.append(np.sum(significant_indices) / len(indices_in_category) * 100)
            not_significant_percentages.append(np.sum(not_significant_indices) / len(indices_in_category) * 100)
        else:
            agreement_percentages.append(None)
            significant_percentages.append(None)
            not_significant_percentages.append(None)
    return categories, agreement_percentages, significant_percentages, not_significant_percentages

def find_place_field(rate_map, place_field_thresh=0.2, noise_thresh=0.2):
    """
    Identify contiguous regions of high firing rates above a threshold, and remove noisy regions below a threshold.
    
    Parameters:
    - rate_map: 2D array representing the spatial firing rate map.
    - place_field_thresh: Fraction of the peak firing rate used for thresholding (e.g., 0.2).
    - noise_thresh: Threshold below which regions are considered noise and excluded.
    
    Returns:
    - place_field_mask: Binary map of the identified place fields (True for regions above the threshold).
    - place_bins_passed_thresh: Number of bins above the place field threshold.
    - labeled_fields: 2D array where each contiguous region above the threshold is labeled with a unique integer.
    - num_fields: Number of contiguous place fields found.
    - peak_rate: Peak firing rate used for thresholding.
    """
    # Find the peak firing rate in the rate map
    peak_rate = np.max(rate_map)
    peak_loc = np.argmax(rate_map)
    
    # Define the threshold value (fraction of peak firing rate)
    place_field_thresh_value = place_field_thresh * peak_rate
    noise_thresh_value = noise_thresh*peak_rate
    
    # Create a binary mask of regions with firing rates above the threshold
    place_field_mask = rate_map >= place_field_thresh_value
    
    # Apply the noise threshold: Remove regions where firing rate is below the noise threshold
    noise_mask = rate_map < noise_thresh_value
    
    
    # Count how many bins (or regions) passed the combined threshold
    num_place_field_bins = place_field_mask.sum()
    num_noise_bins = noise_mask.sum()
    
    # Label contiguous regions above the threshold
    labeled_place_fields, num_place_fields = label(place_field_mask)
    
    # Return all the relevant information
    return num_place_field_bins,num_noise_bins, labeled_place_fields, num_place_fields, peak_rate,peak_loc


def find_place_location(trial_place_bins, peak_loc, tolerance=1):
    max_locs = []  # List to store the indices of max values
    max_values = []  # List to store the max values
          
    for indx, trial in enumerate(trial_place_bins):
        # Find the index of the maximum value in the trial
        max_index = np.argmax(trial)
        max_value = np.max(trial)
        
        # Store the results
        max_locs.append(max_index)
        max_values.append(max_value)

    max_locs = np.array(max_locs)
    max_values = np.array(max_values)

    # Compute the number of max_locs close to the peak_loc within a given tolerance
    close_to_peak = np.abs(max_locs - peak_loc) <= tolerance
    num_close_to_peak = np.sum(close_to_peak)
    percentage_num_close_to_peak = num_close_to_peak/len(trial_place_bins)
    
    return max_locs, max_values, num_close_to_peak,percentage_num_close_to_peak