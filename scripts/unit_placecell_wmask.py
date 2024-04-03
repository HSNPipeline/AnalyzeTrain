"""Run analyses across units."""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from os import path as ospath
from pathlib import Path

from convnwb.io import load_nwbfile,make_session_name,get_files, save_json
from convnwb.utils.log import print_status
from convnwb.io.utils import get_files, file_in_list
from convnwb.utils.run import catch_error

from spiketools.measures.spikes import compute_firing_rate
from spiketools.plts.spatial import plot_position_by_time, plot_position_1d, plot_positions, plot_heatmap,create_heatmap_title
from spiketools.spatial.information import compute_spatial_information
from spiketools.plts.spikes import plot_firing_rates
from spiketools.plts.trials import plot_rasters
from spiketools.stats.shuffle import shuffle_spikes
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
from spiketools.spatial.place import compute_place_bins, compute_trial_place_bins
from spiketools.stats.permutations import compute_surrogate_stats
from spiketools.plts.style import drop_spines
# Import settings from local file
from settings import RUN, PATHS, UNITS
# Local imports
import sys
sys.path.append('../code')
from plts import plot_task_structure,plot_positions_with_speed
from utils import group_array_by_key
from reports import create_sess_str
from group import get_all_session_paths
from models import create_df_place,fit_anova_place
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
    failed_units = []
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
        session_name = make_session_name(experiment, subject, session)
        
        n_trials = len(nwbfile.trials)
        #print('Number of trials: {}'.format(n_trials))
        summary['n_trials'] = str(n_trials)
    
        # Check task time range 
        task_range = [nwbfile.trials.start_time[0], nwbfile.trials.stop_time[-1]]
        task_len = convert_sec_to_min(task_range[1]-task_range[0])
        #print('Task length: {:1.2f} minutes'.format(task_len))
        summary['task_length'] = np.round(task_len,2)
        
        # Get position data
        pos = nwbfile.acquisition['position']['player_position']
        ptimes = pos.timestamps[:]
        positions = pos.data[:]
        
        # Compute position ranges
        x_min_track, x_max_track = compute_pos_ranges(positions)
        track_range = [x_min_track,x_max_track]
        track_length = x_max_track-x_min_track
        #print('Track Range: ', track_range)
        #print('Track Length: ', track_length)

        # Binning 
        num_bins =40
        bins = np.linspace(x_min_track, x_max_track, num_bins+1)
        bin_edges = compute_bin_edges(positions, num_bins)
        bin_width = compute_bin_width(bin_edges)
        #print('Number of bins: ', num_bins)
        #print('Bin widths: {:1.2f}'.format(bin_width))


        # Get speed data 
        speed_thresh = 2
        speed = nwbfile.processing['position_measures']['speed'].data[:]

        # Get move data
        move_start= nwbfile.trials.movement_start_time[:]
        move_end= nwbfile.trials.movement_stop_time[:]
        
        # Get the trial structure based on positions 
        trial_start,trial_end = get_trial_structure(ptimes, positions,mini=-25,maxi = 25, dist = 100)
        min_trials_len = np.min([len(move_start),len(trial_start)])

        # Compute occupancy with spatial mask 
        stoping_zone = 3
        starting_zone= 1
        x_min = x_min_track+starting_zone*1.72
        x_max = x_max_track-stoping_zone*1.72
        move_bins = num_bins-stoping_zone-starting_zone
        track_range = [x_min,x_max]

        move_occ = compute_occupancy(positions, ptimes, move_bins,min_speed=speed_thresh,area_range = track_range)

        ## ANALYZE UNITS

#         # Get unit information
        n_units = len(nwbfile.units)
        #print('Number of units:',n_units)

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
            
            # Extract spikes during movement periods
            spikes_move = epoch_spikes_by_range(spikes, move_start, move_end, reset=False)
            move_spikes_all = np.concatenate(spikes_move).ravel()

            # Filter position data based on starting and stopping zone
            filt_ptimes = ptimes[(positions < x_max) & (positions > x_min)]
            filt_positions = positions[(positions < x_max) & (positions > x_min)]

            filt_spike_pos = []
            spike_t = []
            for ind in range(len(spikes_move)): 
                m_spikes = spikes_move[ind]
                filt_spike_pos.append(get_values_by_times(filt_ptimes, filt_positions, m_spikes,time_threshold = .1))
                spike_t.append(get_values_by_times(filt_ptimes,filt_ptimes,m_spikes,time_threshold = .1))
            spike_times_all = np.concatenate(spike_t).ravel()
            filt_spike_pos_all = np.concatenate(filt_spike_pos).ravel()

            ## Compute measures 

            ## Compute firing rate 
            m_bins = np.linspace(x_min, x_max, move_bins+1)
            move_sc = compute_bin_counts_pos(filt_spike_pos_all, move_bins)
            move_fr = move_sc / move_occ
            
            try:
                ## Spatial Info 
                spike_info = compute_spatial_information(move_fr, move_occ)
                print(f'Spatial_Info: {np.round(spike_info,3)}')

                move_trial_occupancy = compute_trial_occupancy(positions, ptimes, move_bins, move_start, move_end, area_range=track_range)

                 # ANOVA
                move_trial_place_bins = compute_trial_place_bins(spikes, positions, ptimes, move_bins, move_start, move_end, time_threshold=.1, trial_occupancy=move_trial_occupancy, area_range=track_range, flatten=True)
                move_df = create_df_place(move_trial_place_bins)
                move_f_val = fit_anova_place(move_df)
                print('F value: ', round(move_f_val,3))
            except ValueError as error_that_happend:  # Replace SpecificException with the actual exception type you expect
                #catch_error(UNITS['CONTINUE_ON_FAIL'], name, PATHS['RESULTS'] / 'units' / 'zFailed', RUN['VERBOSE'], f'issue running unit #: \t{name}')
                failed_units.append(f'{session_name}_unit{unit_ind}')
                #print(f'{session_name}_unit{unit_ind} failed')
                print(failed_units)
            # json_path = '/Users/weijiazhang/Data/Train/report/unit_place_cell_wmask_report/00_failed_units.json'  # Specify the path to your JSON file

            # # Save the failed_units list to a JSON file
            # with open(json_path, 'w') as json_file:
            #     json.dump(failed_units, json_file)






            # try:
            #     ## Spatial Info 
            #     spike_info = compute_spatial_information(move_fr, move_occ)
            #     print(f'Spatial_Info: {np.round(spike_info,3)}')

            #     move_trial_occupancy = compute_trial_occupancy(positions, ptimes, move_bins, move_start, move_end,area_range = track_range)

            #     # ANOVA
            #     move_trial_place_bins = compute_trial_place_bins(spikes, positions, ptimes,move_bins, move_start, move_end,time_threshold = .1,
            #                                                 trial_occupancy=move_trial_occupancy, area_range = track_range,flatten=True)
                
            #     move_df = create_df_place(move_trial_place_bins)
            #     print(move_df.head())
            #     move_f_val = fit_anova_place(move_df)
            #     #print(move_f_val)
            #     print('F value: ', round(move_f_val,3))
            # except Exception as excp:

            #     catch_error(UNITS['CONTINUE_ON_FAIL'], name, PATHS['RESULTS'] / 'units' / 'zFailed',
            #                  RUN['VERBOSE'], 'issue running unit #: \t{}')

            # ## Shuffle 
            # n_surrogates = 100
            # times_shuffle = shuffle_spikes(move_spikes_all, 'isi', n_shuffles=n_surrogates)

            # ## Shuffle Spike Info 
            # surrs = np.zeros(n_surrogates)
            # for ind, stimes in enumerate(times_shuffle):
            #     spike_p = get_values_by_times(filt_ptimes, filt_positions, stimes,time_threshold = .1)
            #     spike_counts = compute_bin_counts_pos(spike_p, move_bins)
            #     bin_firing = spike_counts / move_occ
            #     surrs[ind] = compute_spatial_information(bin_firing, move_occ, normalize=True)

            # ## Shuffle ANOVA
            # surrogates = np.zeros(n_surrogates)
            # for ind, stimes in enumerate(times_shuffle):
            #     trial_place_bins = compute_trial_place_bins(stimes, filt_positions, filt_ptimes, move_bins, move_start, move_end,
            #                                 trial_occupancy=move_trial_occupancy,time_threshold = .1, flatten=True)
            #     surrogates[ind] = fit_anova_place(create_df_place(trial_place_bins))

            # Initialize figure with grid layout
            grid = make_grid(6, 5, wspace=0.3, hspace=.6, figsize=(30, 25),
                            width_ratios=[.4,.4, 1, 1, 1],
                            title=f'Place Cells W Spatial Mask: {session_name} - unit_{unit_ind}')

            plot_heatmap(move_occ, title=create_heatmap_title('Occupancy-',move_occ), ax = get_grid_subplot(grid, slice(0, 1), slice(2, 5)))

            ax = get_grid_subplot(grid, slice(1, 2), slice(2, 5))
            ax.eventplot(spikes, lineoffsets=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Spike times')
            ax.set_title(f'Unit {unit_ind}')

            ax = get_grid_subplot(grid, slice(2, 3), slice(2, 5))
            ax.eventplot(move_spikes_all, lineoffsets=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Spike times')
            ax.set_title(f'Movement Spikes Unit_{unit_ind}')

            plot_position_by_time(ptimes,positions,spike_times_all,filt_spike_pos_all , ax = get_grid_subplot(grid, slice(3, 4), slice(2, 5)))
            plt.title(f' Unit {unit_ind}: Spike Position by Time')

            plot_positions_with_speed(ptimes, positions, speed,speed_thresh,ax = get_grid_subplot(grid, slice(4, 5), slice(2, 5)))
            plt.title(f' Positions with speed thresh: {speed_thresh}')


            plot_position_by_time(ptimes, positions, ax=get_grid_subplot(grid, slice(5,6), slice(2, 5)))
            for start in trial_start:
                plt.axvline(x=start, color='grey', alpha=0.4, linestyle='--', label='Trial Start' if start == trial_start[0] else "")
            for end in trial_end:
                plt.axvline(x=end, color='black', alpha=0.4, linestyle='--', label='Trial End' if end == trial_end[0] else "")
            plt.title('Positions with Trial Start and End Times')
            plt.xlabel('Time')
            plt.ylabel('Position')
            plt.legend()


            plot_rasters(filt_spike_pos,ax=get_grid_subplot(grid,  slice(0,2), slice(0,2)), vline=None, show_axis=True, title='Move Positions')
            drop_spines(['top', 'right'],ax=get_grid_subplot(grid,  slice(0,2), slice(0,2)))


            ax = get_grid_subplot(grid, 2,slice(0,2))
            ax.plot( m_bins[:-1],move_fr[:], 'k', lw=2)
            ax.set_xlabel('Position on Track (cm)')
            ax.set_ylabel('Firing Rate')
            ax.spines['top'].set_visible(False)  # Hide the top spine
            ax.spines['right'].set_visible(False)

            # Compute statistics on the surrogates
            # p_val, z_score = compute_surrogate_stats(spike_info, surrs, verbose=True,plot = True, ax =get_grid_subplot(grid,  3, slice(0,2)))
            # plt.title(f'Spike Info:{np.round(spike_info,3)} ')
            # # Compute statistics on the surrogates
            # p_val, z_score = compute_surrogate_stats(move_f_val, surrogates, verbose=True,plot = True, ax =get_grid_subplot(grid,  4, slice(0,2)))
            # plt.title(f'ANOVA f-val:{np.round(move_f_val,3)}')
            plt.savefig(f'/Users/weijiazhang/Data/Train/report/unit_place_cell_wmask_report/{session_name}_unit_{unit_ind}_report.pdf')
            plt.close
    json_path = '/Users/weijiazhang/Data/Train/report/unit_place_cell_wmask_report/00_failed_units.json'  # Specify the path to your JSON file

            # Save the failed_units list to a JSON file
    with open(json_path, 'w') as json_file:
        json.dump(failed_units, json_file)
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
