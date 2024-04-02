"""Run analyses across units."""
from convnwb.io import load_nwbfile, make_session_name
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
sns.set_context('talk')

from convnwb.io import load_nwbfile
from convnwb.io import get_files, save_json
from convnwb.utils.log import print_status

from spiketools.stats.shuffle import shuffle_spikes
from spiketools.plts.trials import plot_rasters, create_raster_title
from spiketools.plts.trials import plot_rate_by_time
from spiketools.plts.spatial import plot_position_by_time, plot_position_1d, plot_positions, plot_heatmap,create_heatmap_title
from spiketools.utils.timestamps import convert_sec_to_min, sum_time_ranges
from spiketools.plts.data import plot_bar, plot_hist, plot_text, plot_barh,plot_lines
from spiketools.plts.utils import make_grid, get_grid_subplot
from spiketools.spatial.utils import compute_pos_ranges, compute_bin_width
from spiketools.measures.trials import compute_segment_frs
from spiketools.measures.spikes import compute_firing_rate
from spiketools.plts.spatial import plot_position_by_time,plot_heatmap,create_heatmap_title
from spiketools.utils.extract import get_range
from spiketools.utils.epoch import epoch_spikes_by_range, epoch_spikes_by_event
from spiketools.plts.utils import save_figure
from spiketools.plts.data import plot_barh
from spiketools.plts.trials import plot_rasters
from spiketools.plts.annotate import add_vlines,add_box_shades
from spiketools.utils.extract import (get_range, get_values_by_time_range,
                                      get_values_by_times, threshold_spikes_by_values)
from spiketools.spatial.occupancy import compute_occupancy,compute_trial_occupancy, compute_bin_edges,compute_bin_counts_pos, normalize_bin_counts
from spiketools.utils.extract import get_range, get_values_by_time_range, get_values_by_times,get_inds_by_times
from spiketools.stats.trials import compare_pre_post_activity
from spiketools.measures.trials import compute_trial_frs

from spiketools.plts.style import drop_spines
from spiketools.plts.annotate import color_pvalue
# Import settings from local file
from settings import RUN, PATHS, UNITS
# Local imports
import sys
sys.path.append('../code')
from plts import plot_spikes_trial,plot_task_structure,plot_positions_with_speed
from utils import group_array_by_key,select_movement
from reports import create_sess_str
from trial import get_trial_structure
from group import get_all_session_paths

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
        # Define bins to use
        num_bins = 40

        # Define minimum occupancy
        min_occ = 1
        speed_thresh = 5e-6
        time_thresh = 0.25

        # Set the time range to analyze
        trial_range = [-1, 1]
        pre_window = [-1, 0]
        post_window = [0, 1]

        t_bin = 0.25

        # Shuffle settings
        shuffle_approach = 'BINCIRC'
        n_surrogates = 25

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

        # Get object related information 
        obj = nwbfile.trials.object[:]
        # Get the object positions 
        obj_pos = nwbfile.trials.object_position[:]     
        # Get subject response position and time 
        res_pos = nwbfile.trials.response_position[:]
        res_time = nwbfile.trials.response_time[:]

        # Individual Object Location
        obj_loc = group_array_by_key(obj, obj_pos)
        barrel_loc = np.array(obj_loc['barrel'][:])
        box_loc = np.array(obj_loc['box'][:])
        desk_loc= np.array(obj_loc['desk'][:])
        bench_loc = np.array(obj_loc['bench'][:])

        # Subject Response Location
        res_loc = group_array_by_key(obj, res_pos)
        res_barrel_loc = np.array(res_loc['barrel'][:])
        res_box_loc= np.array(res_loc['box'][:])
        res_desk_loc = np.array(res_loc['desk'][:])
        res_bench_loc = np.array(res_loc['bench'][:])

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

            all_responses = epoch_spikes_by_event(spikes, res_time, trial_range)
            avg_pre, avg_post, t_val, p_val = compare_pre_post_activity(all_responses, pre_window, post_window)
            bin_times, all_frs = compute_trial_frs(all_responses, 0.25, time_range=[-1,1])


            # Get grouped trial spikes 
            barrel_t = epoch_spikes_by_event(spikes, res_barrel_loc, trial_range)
            box_t = epoch_spikes_by_event(spikes, res_box_loc, trial_range)
            bench_t = epoch_spikes_by_event(spikes, res_bench_loc, trial_range)
            desk_t = epoch_spikes_by_event(spikes, res_desk_loc, trial_range)

            # Get grouped trial results
            results = {}
            avg_pre_barrel, avg_post_barrel, results['t_val_barrel'],results['p_val_barrel'] = compare_pre_post_activity(barrel_t, pre_window, post_window)
            avg_pre_box, avg_post_box, results['t_val_box'], results['p_val_box']= compare_pre_post_activity(box_t, pre_window, post_window)
            avg_pre_bench, avg_post_bench, results['t_val_bench'],results['p_val_bench'] = compare_pre_post_activity(bench_t, pre_window, post_window)
            avg_pre_desk, avg_post_desk, results['t_val_desk'],results['p_val_desk'] = compare_pre_post_activity(desk_t, pre_window, post_window)
            
            # Compute firing rates
            bin_times, barrel_frs = compute_trial_frs(barrel_t, t_bin, time_range=[-1,1])
            bin_times, box_frs = compute_trial_frs(box_t, t_bin, time_range=[-1,1])
            bin_times, bench_frs = compute_trial_frs(bench_t, t_bin, time_range=[-1,1])
            bin_times, desk_frs = compute_trial_frs(desk_t, t_bin, time_range=[-1,1])

            # Encoding and recall trials
            barrel_encode = barrel_t[:2]
            barrel_recall = barrel_t[2:]

            box_encode = box_t[:2]
            box_recall = box_t[2:]

            bench_encode = bench_t[:2]
            bench_recall = bench_t[2:]

            desk_encode = desk_t[:2]
            desk_recall = desk_t[2:]

            all_encode =  barrel_encode + box_encode + bench_encode + desk_encode
            all_recall =  barrel_recall + box_recall + bench_recall + desk_recall

            # All encoding trials 
            avg_pre_encode, avg_post_encode, t_val_encode, p_val_encode = compare_pre_post_activity(all_encode, pre_window, post_window)
            bin_times, encode_frs = compute_trial_frs(all_encode, t_bin, time_range=[-1,1])

            # All recall trials
            avg_pre_recall, avg_post_recall, t_val_recall, p_val_recall = compare_pre_post_activity(all_recall, pre_window, post_window)
            bin_times, recall_frs = compute_trial_frs(all_recall, t_bin, time_range=[-1,1])

            results_encode = {}
            fr_pre_barrel_encode, fr_post_barrel_encode, results_encode['t_val_barrel'],results_encode['p_val_barrel'] = compare_pre_post_activity(barrel_encode, pre_window, post_window)
            fr_pre_box_encode, fr_post_box_encode, results_encode['t_val_box'], results_encode['p_val_box']= compare_pre_post_activity(box_encode, pre_window, post_window)
            fr_pre_bench_encode, fr_post_bench_encode, results_encode['t_val_bench'],results_encode['p_val_bench'] = compare_pre_post_activity(bench_encode, pre_window, post_window)
            fr_pre_desk_encode, fr_post_desk_encode, results_encode['t_val_desk'],results_encode['p_val_desk'] = compare_pre_post_activity(desk_encode, pre_window, post_window)

            results_recall = {}
            fr_pre_barrel_recall, fr_post_barrel_recall, results_recall['t_val_barrel'],results_recall['p_val_barrel'] = compare_pre_post_activity(barrel_recall, pre_window, post_window)
            fr_pre_box_recall, fr_post_box_recall, results_recall['t_val_box'], results_recall['p_val_box']= compare_pre_post_activity(box_recall, pre_window, post_window)
            fr_pre_bench_recall, fr_post_bench_recall, results_recall['t_val_bench'],results_recall['p_val_bench'] = compare_pre_post_activity(bench_recall, pre_window, post_window)
            fr_pre_desk_recall, fr_post_desk_recall, results_recall['t_val_desk'],results_recall['p_val_desk'] = compare_pre_post_activity(desk_recall, pre_window, post_window)

            # Computes continuous firing rate across encoding trials 
            bin_times, barrel_frs_encode = compute_trial_frs(barrel_encode, t_bin, time_range=[-1,1])
            bin_times, box_frs_encode = compute_trial_frs(box_encode, t_bin, time_range=[-1,1])
            bin_times, bench_frs_encode = compute_trial_frs(bench_encode, t_bin, time_range=[-1,1])
            bin_times, desk_frs_encode = compute_trial_frs(desk_encode, t_bin, time_range=[-1,1])

            # Computes continuous firing rate across recall trials 
            bin_times, barrel_frs_recall = compute_trial_frs(barrel_recall, t_bin, time_range=[-1,1])
            bin_times, box_frs_recall = compute_trial_frs(box_recall, t_bin, time_range=[-1,1])
            bin_times, bench_frs_recall = compute_trial_frs(bench_recall, t_bin, time_range=[-1,1])
            bin_times, desk_frs_recall = compute_trial_frs(desk_recall, t_bin, time_range=[-1,1])

            # Initialize figure with grid layout
            grid = make_grid(8, 6, wspace=0.3, hspace=0.4, figsize=(30, 35),
                            width_ratios=[1, 1, .5, .5, .5,.5],
                            title=f'Unit Stim Response: {nwbfilename} - unit_{unit_ind}')

            plot_hist(nwbfile.trials['response_error'].data[:], title='Response Error',ax=get_grid_subplot(grid, slice(0, 2), slice(1, 2)))

            plot_position_1d(positions, [barrel_loc,box_loc,desk_loc,bench_loc], ax=get_grid_subplot(grid, slice(0, 1), slice(0, 1)))
            plt.title('object positions')
            #plt.legend(['barrel', 'box', 'desk', 'bench'], loc='upper left', frameon=False, fontsize='large')

            plot_position_1d(positions, [res_barrel_loc, res_box_loc, res_desk_loc, res_bench_loc],
                            title='Response Positions', ax=get_grid_subplot(grid, slice(1, 2), slice(0, 1)))
            #plt.legend(['barrel', 'box', 'desk', 'bench'], loc='upper left', frameon=False, fontsize='large')


            plot_rasters(all_responses,  xlim=trial_range, vline=0,title = create_raster_title('All-Responses', avg_pre, avg_post,
                                                t_val, p_val),
                        title_color=color_pvalue(p_val), title_fontsize=14,
                        ax=get_grid_subplot(grid, slice(2, 3), slice(0, 2)))
            plot_rate_by_time(bin_times, all_frs.mean(0), shade='sem',ax=get_grid_subplot(grid, slice(3, 4), slice(0, 2)))
            add_vlines(0, ax=get_grid_subplot(grid, slice(3, 4), slice(0, 2)), color='green', lw=2.5, alpha=0.5)
            drop_spines(['top', 'right',],get_grid_subplot(grid, slice(3, 4), slice(0, 2)))


            # Encoding 
            plot_rasters(all_encode,  xlim=trial_range, vline=0, 
                        colors= 'blue',title = create_raster_title('All-Encode', avg_pre_encode, avg_post_encode,t_val_encode, p_val_encode),
                        title_color=color_pvalue(p_val), title_fontsize=14,
                        ax=get_grid_subplot(grid, slice(4, 5), slice(0, 2)))
            plot_rate_by_time(bin_times, encode_frs.mean(0), shade='sem',ax=get_grid_subplot(grid, slice(5, 6), slice(0, 2)), colors='blue')
            add_vlines(0, ax=get_grid_subplot(grid, slice(5, 6), slice(0, 2)), color='green', lw=2.5, alpha=0.5)
            drop_spines(['top', 'right',],get_grid_subplot(grid, slice(5, 6), slice(0, 2)))

            # Recall
            plot_rasters(all_recall,  xlim=trial_range, vline=0, 
                        colors= 'black',title = create_raster_title('All-Recall', avg_pre_recall, avg_post_recall,
                                                t_val_recall, p_val_recall),
                        title_color=color_pvalue(p_val), title_fontsize=14,
                        ax=get_grid_subplot(grid, slice(6,7), slice(0, 2)))
            plot_rate_by_time(bin_times, recall_frs.mean(0), shade='sem',ax=get_grid_subplot(grid, slice(7, 8), slice(0, 2)), colors='black')
            add_vlines(0, ax=get_grid_subplot(grid, slice(7, 8), slice(0, 2)), color='green', lw=2.5, alpha=0.5)
            drop_spines(['top', 'right',], ax=get_grid_subplot(grid, slice(7, 8), slice(0, 2)))


            # Barrel 
            plot_rasters(barrel_t,  xlim=trial_range, vline=0, 
                        colors= 'orange',title = create_raster_title('Barrel', avg_pre_barrel, avg_post_barrel,
                                                results['t_val_barrel'], results['p_val_barrel']),
                        title_color=color_pvalue(results['p_val_barrel']), title_fontsize=14,
                        ax=get_grid_subplot(grid, slice(0, 1), slice(2, 4)))
            plot_rate_by_time(bin_times, barrel_frs.mean(0), shade='sem',ax=get_grid_subplot(grid, slice(1, 2), slice(2, 4)), colors='orange')
            add_vlines(0, ax=get_grid_subplot(grid, slice(1, 2), slice(2, 4)), color='green', lw=2.5, alpha=0.5)
            drop_spines(['top', 'right',],get_grid_subplot(grid, slice(1, 2), slice(2, 4)))


            # Barrel - Encoding vs Recall 
            plot_rasters([barrel_encode,barrel_recall], xlim=trial_range, vline=0, colors= ['blue','black'],title = 'Barrel',
                        ax=get_grid_subplot(grid, slice(0, 1), slice(4,6)))
            plot_rate_by_time(bin_times, [barrel_frs_encode.mean(0),barrel_frs_recall.mean(0)], shade='sem', 
                            ax=get_grid_subplot(grid, slice(1, 2), slice(4,6)), colors=['blue','black'])
            add_vlines(0, ax=get_grid_subplot(grid, slice(1, 2), slice(4,6)), color='green', lw=2.5, alpha=0.5)
            drop_spines(['top', 'right'],get_grid_subplot(grid, slice(1, 2), slice(4,6)))

            # Box 
            plot_rasters(box_t,  xlim=trial_range, vline=0, 
                        colors= 'red',title = create_raster_title('Box', avg_pre_box, avg_post_box,results['t_val_box'], results['p_val_box']),
                        title_color=color_pvalue(results['p_val_box']), title_fontsize=14,
                        ax=get_grid_subplot(grid, slice(2, 3), slice(2, 4)))
            plot_rate_by_time(bin_times, box_frs.mean(0), shade='sem', ax=get_grid_subplot(grid, slice(3, 4), slice(2, 4)), colors='red')
            add_vlines(0, ax=get_grid_subplot(grid, slice(3, 4), slice(2, 4)), color='green', lw=2.5, alpha=0.5)
            drop_spines(['top', 'right',],get_grid_subplot(grid, slice(3, 4), slice(2, 4)))
            # Box - Encoding vs Recall 
            plot_rasters([box_encode,box_recall], xlim=trial_range, vline=0, colors= ['blue','black'],title = 'Box',
                        ax=get_grid_subplot(grid, slice(2, 3), slice(4,6)))

            plot_rate_by_time(bin_times, [box_frs_encode.mean(0),box_frs_recall.mean(0)], shade='sem', 
                            ax=get_grid_subplot(grid, slice(3, 4), slice(4,6)), colors=['blue','black'])
            add_vlines(0, ax=get_grid_subplot(grid, slice(3, 4), slice(4,6)), color='green', lw=2.5, alpha=0.5)
            drop_spines(['top', 'right'],get_grid_subplot(grid, slice(3, 4), slice(4,6)))

            # Bench 
            plot_rasters(bench_t,  xlim=trial_range, vline=0, 
                        colors= 'blue',title = create_raster_title('Bench', avg_pre_bench, avg_post_bench,results['t_val_bench'], results['p_val_bench']),
                        title_color=color_pvalue(results['p_val_bench']), title_fontsize=14,
                        ax=get_grid_subplot(grid, slice(4, 5), slice(2,4)))
            plot_rate_by_time(bin_times, bench_frs.mean(0), shade='sem', 
                            ax=get_grid_subplot(grid, slice(5, 6), slice(2,4)), colors='blue')
            add_vlines(0, ax=get_grid_subplot(grid, slice(5,6), slice(2,4)), color='green', lw=2.5, alpha=0.5)
            drop_spines(['top', 'right',],ax=get_grid_subplot(grid, slice(5, 6), slice(2,4)))

            # Bench-Encoding and Recall 
            plot_rasters([bench_encode,bench_recall], xlim=trial_range, vline=0, 
                        colors= ['blue','black'],title = 'Bench',
                        ax=get_grid_subplot(grid, slice(4,5), slice(4,6)))
            plot_rate_by_time(bin_times, [bench_frs_encode.mean(0),bench_frs_recall.mean(0)], shade='sem', 
                            ax=get_grid_subplot(grid, slice(5,6), slice(4,6)), colors=['blue','black'])
            add_vlines(0,ax=get_grid_subplot(grid, slice(5,6), slice(4,6)), color='green', lw=2.5, alpha=0.5)
            drop_spines(['top', 'right'],get_grid_subplot(grid, slice(5,6), slice(4,6)))

            # Desk 
            plot_rasters(desk_t,  xlim=trial_range, vline=0, 
                        colors= 'purple',title = create_raster_title('Desk', avg_pre_desk, avg_post_desk,
                                                results['t_val_desk'], results['p_val_desk']),
                        title_color=color_pvalue(results['p_val_desk']), title_fontsize=14,
                        ax=get_grid_subplot(grid, slice(6, 7), slice(2,4)))

            plot_rate_by_time(bin_times, desk_frs.mean(0), shade='sem', 
                            ax=get_grid_subplot(grid, slice(7, 8), slice(2,4)), colors='purple')
            add_vlines(0, ax=get_grid_subplot(grid, slice(7, 8), slice(2,4)), color='green', lw=2.5, alpha=0.5)
            drop_spines(['top', 'right',],ax=get_grid_subplot(grid, slice(7, 8), slice(2,4)))

            # Desk - Encoding and Recall 
            plot_rasters([desk_encode,desk_recall], xlim=trial_range, vline=0, 
                        colors= ['blue','black'],title = 'Desk',
                        ax=get_grid_subplot(grid, slice(6, 7), slice(4,6)))

            plot_rate_by_time(bin_times, [desk_frs_encode.mean(0),desk_frs_recall.mean(0)], shade='sem', ax=get_grid_subplot(grid, slice(7, 8), slice(4,6)), colors=['blue','black'])
            add_vlines(0, ax=get_grid_subplot(grid, slice(7, 8), slice(4,6)), color='green', lw=2.5, alpha=0.5)
            drop_spines(['top', 'right'],ax=get_grid_subplot(grid, slice(7, 8), slice(4,6)))


            plt.savefig(f'/Users/weijiazhang/Data/Train/report/unit_stim_report/{nwbfilename}_unit_{unit_ind}_report.pdf')
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
