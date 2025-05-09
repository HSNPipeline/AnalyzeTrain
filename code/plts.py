"""Plotting functions for Train Analysis."""

import numpy as np
import matplotlib.pyplot as plt



from spiketools.plts.utils import check_ax
from spiketools.plts.style import set_plt_kwargs


from spiketools.plts.task import plot_task_structure as _plot_task_structure
from spiketools.plts.trials import plot_rasters
from spiketools.plts.data import plot_barh
from spiketools.plts.spatial import create_heatmap_title
from spiketools.plts.utils import check_ax, make_grid, get_grid_subplot
from spiketools.plts.style import set_plt_kwargs, drop_spines
from spiketools.plts.annotate import add_vlines, add_box_shades


###################################################################################################
###################################################################################################

def plot_task_structure(trials, ax=None, **plt_kwargs):
    """Plot the task structure for Treasure Hunt.

    Parameters
    ----------
    trials : pynwb.epoch.TimeIntervals
        The TreasureHunt trials structure from a NWB file.
    """

    _plot_task_structure([[trials.cue_start_time[:], trials.cue_stop_time[:]],
                          [trials.movement_start_time[:], trials.movement_stop_time[:]],
                          [trials.fixation_start_time[:], trials.fixation_stop_time[:]],
                          [trials.feedback_start_time[:],trials.feedback_stop_time[:]]],
                         [trials.start_time[:], trials.stop_time[:],trials.response_time[:]],
                         range_colors=['green', 'orange', 'purple','blue'],
                         line_colors=['red', 'black','grey'],
                         event_kwargs={'lw' : 1.25},
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
                 title=create_heatmap_title('{}'.format(title), frs))

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
        
        
def plot_positions_with_speed(raw_ptimes, positions, speed, speed_thresh,ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 5))
    filtered_speed_indx = np.where(speed <= speed_thresh)[0]
    
    # Plot all positions on the provided ax with a basic color (e.g., blue)
    ax.plot(raw_ptimes, positions, alpha=0.5)

    # Highlight positions with filtered speeds using a different color (e.g., red)
    ax.scatter(raw_ptimes[filtered_speed_indx], positions[filtered_speed_indx],s=5, color='green',alpha=0.5, label='Filtered Positions')

    # Add labels, legend to the ax
    ax.set_xlabel('Time')
    ax.set_ylabel('Positions')
    ax.set_title('Positions with Speed Threshold')


@set_plt_kwargs
def plot_percentages(steps,percentages,ax= None,**plt_kwargs):
    ax = check_ax(ax,figsize = plt_kwargs.pop('figsize',None))

    step_idx = [i for i, x in enumerate(steps) if x is not None]
    cont_steps = [steps[i] for i in step_idx]
    cont_pct = [percentages[i] for i in step_idx]

    ax.plot(cont_steps,cont_pct,marker='o',markersize=5,**plt_kwargs)
    

@set_plt_kwargs
def plot_raster_with_tuning_curve(data, index, num_trials=64,
                                 raster_color='k', curve_color='red', sem_alpha=0.3, ax=None, **plt_kwargs):
    """
    Create a raster plot with tuning curve for a specific neuron.
    
    Parameters:
    -----------
    data : DataFrame
        DataFrame containing spike data and analysis results
    index : int
        Index of the neuron to plot
    num_trials : int, optional
        Number of trials to plot (default: 64)
    raster_color : str or color, optional
        Color for raster plot dots (default: 'k')
    curve_color : str or color, optional
        Color for tuning curve line (default: 'red')
    sem_alpha : float, optional
        Alpha transparency for SEM shading (default: 0.3)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.
    **plt_kwargs : dict
        Additional keyword arguments to pass to the plot function
        
    Returns:
    --------
    ax, ax2 : matplotlib axes objects
    """
    spike_position = data['spike_position'].iloc[index]
    trial_changes = data['trial_changes'].iloc[index]

    F = data['place_anova'].iloc[index]
    SI = data['place_info'].iloc[index]
    n_bins = np.linspace(0, 1, 41)  # Reset to 0-1 percentage range

    place_bins = np.array(data['place_bins'].iloc[index])
    sem = np.array(data['place_sem'].iloc[index])


    ax = check_ax(ax, figsize=plt_kwargs.pop('figsize', None))
    ax2 = ax.twinx() 
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.tick_left()
    
    # Plot raster
    spikes_positions_trials = {}
    for num_trial in range(num_trials):
        # Define trial indices
        trial_idx = [0, *trial_changes, len(spike_position)]
        
        # Extract spike positions for this trial
        if num_trial < len(trial_idx) - 1:
            trial_start = trial_idx[num_trial]
            trial_end = trial_idx[num_trial + 1]
            spikes_positions_trials[num_trial] = spike_position[trial_start:trial_end]
            
            # Plot spike positions for this trial
            if len(spikes_positions_trials[num_trial]) > 0:
                ax.plot(spikes_positions_trials[num_trial], 
                        [num_trial + 1] * len(spikes_positions_trials[num_trial]), 
                        '.', color=raster_color, markersize=5,alpha=0.3)
    
    # Configure axes
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_xticks([])
    ax2.set_ylabel('')
    
    # Plot tuning curve
    ax2.plot(n_bins[:-2], place_bins[:-1], color=curve_color, linewidth=4, label='Mean Value')
    ax2.fill_between(n_bins[:-2], place_bins[:-1] - sem[:-1], place_bins[:-1] + sem[:-1], color=curve_color, alpha=sem_alpha)
    ax2.set_title(f'F = {F:.2f}, SI = {SI:.2f}',fontsize = 30)
    # Clean up spines
    drop_spines(['top', 'right', 'bottom'], ax=ax)
    drop_spines(['top', 'right'], ax=ax2)
    
    return ax, ax2