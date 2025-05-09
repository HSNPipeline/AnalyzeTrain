"""Helper functions for creating reports."""

import numpy as np

###################################################################################################
###################################################################################################
def create_sess_str(session):
    """Create a string representation of the group info."""

    string = '\n'.join([
        '\n',
        'experiment:  {}'.format(session['experiment']),
        'subject:  {}'.format(session['subject']),
        'session:  {}'.format(session['session']),
        'number of trials: {}'.format(session['n_trials']),
        'task length: {}'.format(session['task_length']),
        'number of units: {}'.format(session['n_units'])
    ])

    return string

"""Helper functions for creating reports."""

import numpy as np

from spiketools.measures.spikes import compute_firing_rate


# ###################################################################################################
# ###################################################################################################

## GROUP
def get_significant_percentage(stats, p, increment, threshold=0.05):
    """
    Calculate percentage of significant values within binned ranges.
    
    Parameters:
    -----------
    stats : array-like
        Values to be binned (e.g., spatial information scores)
    p : array-like
        Corresponding p-values
    increment : float
        Bin size
    threshold : float, default=0.05
        Significance threshold
        
    Returns:
    --------
    bin_centers : list
        Center of each bin (None if bin is empty)
    significant_percentages : list
        Percentage of significant values per bin (None if bin is empty)
    bin_edges : list
        Edges of each bin
    """
    # Remove NaN values
    mask = ~np.isnan(stats)
    stats = stats[mask]
    p = p[mask]
    
    max_val = int(np.max(stats)) + 1
    n_bins = int(max_val / increment)
    
    bin_centers = []
    sig_pct = []
    
    for i in range(n_bins):
        lower = i * increment
        upper = (i + 1) * increment
        center = lower + (increment / 2)

        # Find values in this bin
        in_bin = np.where((stats >= lower) & (stats < upper))[0]
        
        if len(in_bin) > 0:
            p_in_bin = p[in_bin]
            sig_count = np.sum(p_in_bin < threshold)
            percentage = (sig_count / len(in_bin)) * 100
            bin_centers.append(center)
        else:
            percentage = None
            bin_centers.append(None)

        sig_pct.append(percentage)
    
    return bin_centers, sig_pct

def get_agreement_percentage(stats, p_values1, p_values2, increment=0.2, threshold=0.05):
    """
    Calculate the percentage of agreement between two sets of p-values across binned statistic values.
    
    This function bins the statistic values and calculates three types of agreement percentages:
    1. Overall agreement (both significant or both not significant)
    2. Agreement on significance (both p-values below threshold)
    3. Agreement on non-significance (both p-values above threshold)
    
    Parameters:
    -----------
    stats : array-like
        Statistical values to bin
    p_values1 : array-like
        First set of p-values corresponding to stats
    p_values2 : array-like
        Second set of p-values corresponding to stats
    increment : float, default=0.2
        Size of each bin
    threshold : float, default=0.05
        Significance threshold for p-values
        
    Returns:
    --------
    bin_centers : list
        Upper bound of each bin (used as bin center)
    agree_pct : list
        Percentage of overall agreement per bin (None if bin is empty)
    sig_pct : list
        Percentage of agreement on significance per bin (None if bin is empty)
    not_sig_pct : list
        Percentage of agreement on non-significance per bin (None if bin is empty)
    """
    # Remove NaN values
    mask = ~np.isnan(stats)
    stats = stats[mask]
    p_values1 = p_values1[mask]
    p_values2 = p_values2[mask]
    
    if len(stats) == 0:
        return [], [], [], []
    
    max_val = int(np.max(stats)) + 1
    n_bins = int(max_val / increment)
    
    bin_centers = []
    agree_pct = []
    sig_pct = []
    not_sig_pct = []
    
    for i in range(n_bins):
        lower = i * increment
        upper = (i + 1) * increment
        center = upper  # Using upper bound as the bin center
        bin_centers.append(center)
        
        # Find values in this bin
        in_bin = np.where((stats >= lower) & (stats < upper))[0]
        
        if len(in_bin) > 0:
            p1_in_bin = p_values1[in_bin]
            p2_in_bin = p_values2[in_bin]
            
            # Calculate agreements
            both_sig = (p1_in_bin < threshold) & (p2_in_bin < threshold)
            both_not_sig = (p1_in_bin >= threshold) & (p2_in_bin >= threshold)
            
            # Total agreement percentage
            n_agree = np.sum(both_sig) + np.sum(both_not_sig)
            agree_percentage = (n_agree / len(in_bin)) * 100
            
            # Store percentages
            agree_pct.append(agree_percentage)
            sig_pct.append((np.sum(both_sig) / len(in_bin)) * 100)
            not_sig_pct.append((np.sum(both_not_sig) / len(in_bin)) * 100)
        else:
            agree_pct.append(None)
            sig_pct.append(None)
            not_sig_pct.append(None)
    
    return bin_centers, agree_pct, sig_pct, not_sig_pct


## UNIT

def create_unit_info(unit):
    """Create a dictionary of unit information."""

    spikes = unit['spike_times'].values[0]

    unit_info = {}

    unit_info['n_spikes'] = len(spikes)
    unit_info['firing_rate'] = compute_firing_rate(spikes)
    unit_info['first_spike'] = spikes[0]
    unit_info['last_spike'] = spikes[-1]
    #unit_info['hemisphere'] = unit['hemisphere'].values[0]
    #unit_info['location'] = unit['location'].values[0]
    unit_info['channel'] = unit['channel'].values[0]

    return unit_info


def create_unit_str(unit_info):
    """Create a string representation of the unit info."""

    string = '\n'.join([
        '# spikes:    {:10d}'.format(unit_info['n_spikes']),
        'firing rate:  {:10.2f}'.format(unit_info['firing_rate']),
        #'location:    {}'.format(unit_info['location']),
        'channel:    {}'.format(unit_info['channel']),
        #'cluster:    {}'.format(unit_info['cluster']),

        'Recording time range: {:5.4f} - {:5.4f}'.format(\
           unit_info['first_spike'], unit_info['last_spike'])
    ])

    return string