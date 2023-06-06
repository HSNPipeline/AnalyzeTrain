"""Helper functions for creating reports."""

###################################################################################################
###################################################################################################

import numpy as np

from spiketools.measures.spikes import compute_firing_rate, compute_presence_ratio

from spiketools.utils.base import count_elements
from spiketools.utils.extract import drop_range
from spiketools.utils.timestamps import convert_sec_to_min



def create_unit_info(unit, uind):
    """Create a dictionary of unit information."""

    spikes = unit.spike_times.data[:]
    unit_info = {}
    unit_info['n_spikes'] = len(spikes)
    unit_info['firing_rate'] = float(compute_firing_rate(spikes))
    unit_info['presence_ratio'] = float(compute_presence_ratio(spikes, 40.0))
    unit_info['first_spike'] = spikes[0]
    unit_info['last_spike'] = spikes[-1]

    return unit_info


def create_unit_str(unit_info):
    
    string = '\n'.join([
        '\n',
        '\n',
        'spikes:   {:5d}'.format(unit_info['n_spikes']),
        'firing rate:  {:5.4f}'.format(unit_info['firing_rate']),
        'prence_ratio: {:5.4f}'.format(unit_info['presence_ratio']),
        'first_spike: {:5.4f}'.format(unit_info['first_spike']),
        'last_spike: {:5.4f}'.format(unit_info['last_spike']),
          
       ])
    return string



