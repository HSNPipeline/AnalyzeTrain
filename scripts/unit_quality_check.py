"""Run analyses across units."""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from os import path as ospath
from pathlib import Path

from convnwb.io import load_nwbfile,make_session_name,get_files, save_json
from convnwb.utils.log import print_status
from convnwb.io.utils import get_files, file_in_list
from convnwb.utils.run import catch_error

from spiketools.measures.spikes import compute_firing_rate
from spiketools.plts.spatial import plot_position_by_time, plot_position_1d, plot_positions, plot_heatmap,create_heatmap_title
from spiketools.plts.spikes import plot_firing_rates
from spiketools.plts.trials import plot_rasters

from spiketools.plts.data import plot_bar, plot_hist, plot_text, plot_barh,plot_lines
from spiketools.plts.utils import make_axes
from spiketools.plts.annotate import add_hlines, add_vlines,add_box_shades

from spiketools.utils.timestamps import convert_sec_to_min, sum_time_ranges
from spiketools.utils.extract import (get_range, get_values_by_time_range,
                                      get_values_by_times, threshold_spikes_by_values)
from spiketools.utils.epoch import epoch_spikes_by_range, epoch_spikes_by_event
from spiketools.utils.data import compute_range
from spiketools.utils.base import count_elements
from spiketools.plts.utils import make_grid, get_grid_subplot

from spiketools.utils.timestamps import convert_sec_to_min
from spiketools.spatial.occupancy import compute_occupancy,compute_trial_occupancy, compute_bin_edges,compute_bin_counts_pos, normalize_bin_counts
from spiketools.spatial.utils import compute_pos_ranges, compute_bin_width

# Import settings from local file
from settings import RUN, PATHS, UNITS
# Local imports
import sys
sys.path.append('../code')
from plts import plot_task_structure,plot_positions_with_speed
from utils import group_array_by_key
from reports import create_sess_str
from group import get_all_session_paths
from trial import get_trial_structure
###################################################################################################
###################################################################################################

def main():
    """Run unit analyses."""

    print_status(RUN['VERBOSE'], '\n\nANALYZING UNIT DATA - {}\n\n'.format(RUN['TASK']), 0)
    base_path = "/Users/weijiazhang/Data/Train/nwb"
    nwbfiles = get_files(base_path, select='nwb')
    print(nwbfiles)
	
   #  # Get list of already generated and failed units, & drop file names
#     output_files = get_files(PATHS['RESULTS'] / 'units',
#                              select='json', drop_extensions=True)
#     failed_files = get_files(PATHS['RESULTS'] / 'units' / 'zFailed',
#                              select='json', drop_extensions=True)

    for nwbfilename in nwbfiles:

        ## DATA LOADING

        # Check and ignore files set to ignore
        if nwbfilename.split('.')[0] in RUN['IGNORE']:
            print_status(RUN['VERBOSE'], '\nSkipping file (set to ignore): {}'.format(nwbfilename), 0)
            continue

        # Print out status
        print_status(RUN['VERBOSE'], '\nRunning unit analysis: {}'.format(nwbfilename), 0)

        # Load NWB file
        nwbfile, io = load_nwbfile(nwbfilename, base_path, return_io=True)

        ## GET DATA

        part_subj = nwbfilename.split('_')
        part_sess = part_subj[-1].split('.')
        experiment = part_subj[-4]
        subject = part_subj[-3]
        session = part_sess[-2]

        print('experiment:',experiment)
        print('Subject: ',subject)
        print(f'session_{session}')

        summary = {
        'experiment' : experiment,
        'subject' : subject,
        'session' : session   
    }
        num_bins = 40
        n_trials = len(nwbfile.trials)
        print('Number of trials: {}'.format(n_trials))
        summary['n_trials'] = str(n_trials)
    
        # Check task time range 
        task_range = [nwbfile.trials.start_time[0], nwbfile.trials.stop_time[-1]]
        task_len = convert_sec_to_min(task_range[1]-task_range[0])
        print('Task length: {:1.2f} minutes'.format(task_len))
        summary['task_length'] = np.round(task_len,2)
        
        # Get position data
        pos = nwbfile.acquisition['position']['player_position']
        ptimes = pos.timestamps[:]
        positions = pos.data[:]
        
        # Get speed data 
        speed_thresh = 2
        speed = nwbfile.processing['position_measures']['speed'].data[:]

        # Compute occupancy 
        occ = compute_occupancy(positions, ptimes, bins = num_bins)
        
        # Get move data
        move_start= nwbfile.trials.movement_start_time[:]
        move_end= nwbfile.trials.movement_stop_time[:]
        
        # Get the trial structure based on positions 
        trial_start,trial_end = get_trial_structure(ptimes, positions,mini=-25,maxi = 25, dist = 100)
        min_trials_len = np.min([len(move_start),len(trial_start)])
    
        

        ## ANALYZE UNITS

#         # Get unit information
        n_units = len(nwbfile.units)
        print('Number of units:',n_units)

        # Loop across all units
        for unit_ind in range(n_units):

            # Initialize output unit file name & output dictionary
            name = session + '_U' + str(unit_ind).zfill(2)
            results = {}

            # Check if unit already run
            if UNITS['SKIP_ALREADY_RUN'] and name in output_files:
                print_status(RUN['VERBOSE'], 'skipping unit (already run): \tU{:02d}'.format(unit_ind), 1)
                continue

            if UNITS['SKIP_FAILED'] and name in failed_files:
                print_status(RUN['VERBOSE'], 'skipping unit (failed): \tU{:02d}'.format(unit_ind), 1)
                continue

            print_status(RUN['VERBOSE'], 'running unit: \t\t\tU{:02d}'.format(unit_ind), 1)

            # Extract spikes for a unit of interest
            spikes = nwbfile.units.get_unit_spike_times(unit_ind)
            
            # Compute Measures
            spikes_move = epoch_spikes_by_range(spikes, move_start, move_end, reset=True)
            move_spikes = []
            move_frs = np.zeros(len(spikes_move))
            move_spike_pos = []
            for ind in range(len(spikes_move)): 
                m_spikes = get_range(spikes, move_start[ind],move_end[ind])
                move_spike_pos.append(get_values_by_times(ptimes, positions, m_spikes))
                move_spikes.append(m_spikes)
                move_frs[ind] = compute_firing_rate(spikes_move[ind],task_range)
                move_spikes_all = np.concatenate(move_spikes).ravel()
            
            
            grid = make_grid(7, 5, wspace=0.4, hspace=0.5, figsize=(30, 25),width_ratios=[0.5, 0.5, 1, 1, 1], title=f'Unit Report: {unit_ind}')
            plot_heatmap(occ, title=create_heatmap_title('Occupancy-',occ), ax=get_grid_subplot(grid, slice(0,1), slice(0, 2)))
            plot_position_by_time(ptimes,positions,ax=get_grid_subplot(grid, slice(0, 1), slice(2, 5)))
            plt.title('Positions')
            
            plot_position_by_time(ptimes, positions, alpha=0.85,ax=get_grid_subplot(grid, slice(1,2), slice(2, 5)))

            for start in move_start:
                plt.axvline(x=start, color='grey', alpha=0.4)
            for end in move_end:
                plt.axvline(x=end, color='black', alpha=0.4)
                plt.title('Positions with Move Start and End Times')
            plt.xlabel('Time')
            plt.ylabel('Position')

			# Plot the difference between movement start and trial start 
            ax=get_grid_subplot(grid, slice(1,2), slice(0, 2))
            ax.plot(trial_start[:min_trials_len]-move_start[:min_trials_len])
            ax.set_title('Diff in trial and move start')

            ax=get_grid_subplot(grid, slice(2,3), slice(0, 2))
            ax.plot(trial_end[:min_trials_len]-move_end[:min_trials_len])
            ax.set_title('Diff in trial and move end')
            
            plot_rasters(move_spikes_all, show_axis=True, 
                 ylabel='movement periods', yticks=[],ax=get_grid_subplot(grid, slice(2,3), slice(2, 5)))
            add_vlines(trial_start,  color='grey',alpha = .5)   # navigation starts
            add_vlines(trial_end,  color='black',alpha = .5)  # navigation stops
            
            spikes = move_spikes_all
            print(spikes.shape)
            x_min_track, x_max_track = compute_pos_ranges(positions)
            track_range = [x_min_track, x_max_track]
            print('Min Track Position: ', x_min_track)

            bins = np.linspace(x_min_track, x_max_track, num_bins+1)
            spike_pos= get_values_by_times(ptimes, positions, spikes)
            spike_counts = compute_bin_counts_pos(spike_pos, num_bins)
            firing_rates = spike_counts / occ
            
            plot_rasters(move_spike_pos,ax=get_grid_subplot(grid, 3, slice(0,2)), vline=None, show_axis=True, title='Move Positions')
            ax = get_grid_subplot(grid, 4,slice(0,2))
            ax.plot( bins[:-1],firing_rates[:], 'k', lw=2)
            ax.set_xlabel('Position on Track (cm)')
            ax.set_ylabel('Firing Rate')
            ax.spines['top'].set_visible(False)  # Hide the top spine
            ax.spines['right'].set_visible(False)
            
            ypos = (np.arange(len(spikes_move))).tolist()
            plot_barh(move_frs, ypos,xlabel="FR",ax=get_grid_subplot(grid, slice(3,5),  slice(2,3)))
            
            plot_rasters(spikes_move, show_axis=True,  xlabel='Spike times',
                 ylabel="Trial number", yticks=range(0, len(spikes_move)),ax =get_grid_subplot(grid, slice(3,5),  slice(3,5)) )
            add_box_shades(move_end-move_start, np.arange(len(spikes_move)), x_range=0.1, y_range=0.5)
            
            plt.savefig(f'/Users/weijiazhang/Data/Train/report/unit_quality_report/{nwbfilename}_unit_{unit_ind}_report.pdf')
#         plt.close
# #             try:
# # 
# #                 ## Compute measures
# #                 ...
# # 
# #                 ## Collect results
# #                 ...
# # 
# # #                 # Save out unit results
# # #                 save_json(results, name + '.json', folder=str(PATHS['RESULTS'] / 'units'))
# # # 
# # #             except Exception as excp:
# # # 
# # # #                 catch_error(UNITS['CONTINUE_ON_FAIL'], name, PATHS['RESULTS'] / 'units' / 'zFailed',
# # # #                             RUN['VERBOSE'], 'issue running unit #: \t{}')

#     print_status(RUN['VERBOSE'], '\n\nCOMPLETED UNIT ANALYSES\n\n', 0)

if __name__ == '__main__':
    main()
