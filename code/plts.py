"""Plotting functions for Train Analysis."""

import numpy as np
import matplotlib.pyplot as plt

from spiketools.plts.task import plot_task_structure as _plot_task_structure
from spiketools.plts.trials import plot_rasters
from spiketools.plts.data import plot_barh
from spiketools.plts.spatial import create_heat_title
from spiketools.plts.utils import check_ax, savefig, make_grid, get_grid_subplot
from spiketools.plts.style import set_plt_kwargs, drop_spines
from spiketools.plts.annotate import add_vlines, add_box_shades, add_hlines
from spiketools.plts.spatial import plot_positions, plot_heatmap
from spiketools.plts.annotate import add_dots
#from spiketools.utils.data import make_row_orientation, smooth_data, compute_range
###################################################################################################
###################################################################################################

def plot_task_structure(trials, ax=None, **plt_kwargs):
    """Plot the task structure for Treasure Hunt.

    Parameters
    ----------
    trials : pynwb.epoch.TimeIntervals
        The TreasureHunt trials structure from a NWB file.
    """

    _plot_task_structure([[trials.hold_start_time[:], trials.hold_stop_time[:]],
                          [trials.movement_start_time[:], trials.movement_stop_time[:]],
                          [trials.fixation_start_time[:], trials.fixation_stop_time[:]],
                          [trials.feedback_start_time[:],trials.feedback_stop_time[:]]],
                         [trials.start_time[:], trials.stop_time[:],trials.response_time[:]],
                         range_colors=['green', 'orange', 'purple','blue'],
                         line_colors=['red', 'black','grey'],
                         line_kwargs={'lw' : 1.25},
                         ax=ax, **plt_kwargs)
    

