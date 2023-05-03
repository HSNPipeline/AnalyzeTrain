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
    
def plot_spikes_trial(spikes, tspikes, movement_spikes, mov_starts, mov_stops, tmov_stops,
                      response_time, frs, title, hlines=None, **plt_kwargs):
    """Plot the spike raster for whole session, navigation periods and individual trials."""

    # Data orgs
    ypos = (np.arange(len(tspikes))).tolist()

    # Initialize grid
    grid = make_grid(3, 2, width_ratios=[5, 1],
                     wspace=0.1, hspace=0.2, figsize=(18, 20))

    # Row 0: spikes across session
    ax0 = get_grid_subplot(grid, 0, slice(0, 2))
    plot_rasters(spikes, ax=ax0, show_axis=True, ylabel='spikes from whole session', yticks=[],
                 title=create_heat_title('{}'.format(title), frs))
    add_vlines(mov_stops, ax=ax0, color='purple')   # navigation starts
    add_vlines(mov_starts, ax=ax0, color='orange')  # navigation stops

    # Row 1: spikes from navigation periods
    ax1 = get_grid_subplot(grid, 1, slice(0, 2))
    plot_rasters(movement_spikes, vline=response_time, show_axis=True, ax=ax1,
                 ylabel='Spikes from movement periods', yticks=[])

    # Row 2: spikes across trials, with bar plot
    ax2 = get_grid_subplot(grid, 2, 0)
    ax2b = get_grid_subplot(grid, 2, 1, sharey=ax2)
    plot_rasters(tspikes, show_axis=True, ax=ax2, xlabel='Spike times',
                 ylabel="Trial number", yticks=range(0, len(tspikes)))
    add_box_shades(tmov_stops, np.arange(len(tspikes)), x_range=0.1, y_range=0.5, ax=ax2)
    plot_barh(frs, ypos, ax=ax2b, xlabel="FR")
    #if hlines:
    #    add_hlines(hlines, ax=ax2, color='green', alpha=0.4)

    for cax in [ax0, ax1, ax2, ax2b]:
        drop_spines(['top', 'right'], cax)

