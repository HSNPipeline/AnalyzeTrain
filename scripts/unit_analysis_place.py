"""Analysis script: Train unit analysis."""

import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore


from convnwb.io import load_nwbfile, save_json
from convnwb.io.utils import get_files, file_in_list
from convnwb.utils.log import print_status
from convnwb.utils.run import catch_error

from spiketools.measures.spikes import compute_isis
from spiketools.measures.trials import compute_trial_frs
from spiketools.spatial.information import compute_spatial_information
from spiketools.spatial.occupancy import (compute_occupancy, compute_trial_occupancy,compute_bin_edges,
                                          compute_bin_counts_pos, normalize_bin_counts)
from spiketools.spatial.place import compute_place_bins, compute_trial_place_bins
from spiketools.spatial.utils import compute_pos_ranges
from spiketools.stats.permutations import compute_surrogate_stats
from spiketools.stats.shuffle import shuffle_spikes
from spiketools.plts.data import plot_text
from spiketools.plts.spikes import plot_waveform, plot_isis
from spiketools.plts.trials import plot_rasters, plot_raster_and_rates
from spiketools.plts.spatial import plot_heatmap, create_heatmap_title
from spiketools.plts.stats import plot_surrogates
from spiketools.plts.annotate import add_vlines, add_hlines,color_pvalue
from spiketools.plts.utils import make_grid, get_grid_subplot, save_figure
from spiketools.plts.style import drop_spines
from spiketools.utils.extract import (get_range, get_values_by_time_range,
                                      get_values_by_times, threshold_spikes_by_values)
from spiketools.utils.epoch import epoch_spikes_by_range, epoch_spikes_by_event
from spiketools.utils.trials import split_trials_by_condition
from spiketools.utils.base import add_key_prefix, combine_dicts
from spiketools.utils.run import create_methods_list

# Import settings from local file
from settings import RUN, PATHS,OCCUPANCY,OCCUPANCY_TRIAL, PLACE,UNITS,METHODS, SURROGATES,BINS,EXCLUSION,RUN_SPECIFIC


# Import local code functions
import sys
sys.path.append('../code')
from models import (create_df_place, fit_anova_place,)

from reports import create_unit_info, create_unit_str
from utils import get_trial_pos_times,get_values_by_times

#from plts import COLORS, splitter_plot

###################################################################################################
###################################################################################################

def main():
    """Run analyses across all units."""

    # Supress ANOVA warnings. ToDo: fix this later
    warnings.filterwarnings("ignore")

    print_status(RUN['VERBOSE'], '\n\nANALYZING UNIT DATA - {}\n\n'.format(RUN['TASK']), 0)

    # Define output folders
    results_folder = PATHS['RESULTS'] / 'units_place'
    reports_folder = PATHS['REPORTS'] / 'units_place'
   
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(reports_folder, exist_ok=True)
    os.makedirs(results_folder / 'zFailed', exist_ok=True)
    # Collect a copy of all settings with a prefixes
    all_settings = [
        add_key_prefix(BINS, 'bins'),
        add_key_prefix(OCCUPANCY, 'occupancy'),
        add_key_prefix(PLACE, 'place'),
        add_key_prefix(SURROGATES, 'surrogates'),
    ]

    # Save out run settings
    save_json(METHODS, 'methods.json', folder=results_folder)
    save_json(combine_dicts([RUN, UNITS]), 'run.json', folder=results_folder)
    save_json(combine_dicts(all_settings), 'settings.json', folder=results_folder)

    # Get the list of NWB files
    nwbfiles = get_files(PATHS['DATA'], select='nwb')

    # Get list of already generated run units, & drop file names
    output_files = get_files(results_folder, select='json', drop_extensions=True)
    failed_files = get_files(results_folder / 'zFailed', select='json', drop_extensions=True)
    
    if RUN_SPECIFIC:
        for cfile in RUN_SPECIFIC:
            assert cfile + '.nwb' in nwbfiles, 'File {} set to run is not available.'.format(cfile)
        print('Running specific files: {}'.format(', '.join(RUN_SPECIFIC)))
        nwbfiles = RUN_SPECIFIC

    for nwbfilename in nwbfiles:

        ###########################################################################################
        ## DATA LOADING

        # Check and ignore files set to ignore
        if file_in_list(nwbfilename, RUN['IGNORE']):
            print_status(\
                RUN['VERBOSE'], '\nSkipping file (set to ignore): {}'.format(nwbfilename), 0)
            continue

        # Print out status
        print_status(\
            RUN['VERBOSE'],'\nRunning unit analysis: {}'.format(nwbfilename), 0)

        # Load NWB file
        nwbfile, io = load_nwbfile(nwbfilename, PATHS['DATA'], return_io=True)
        

        # Get position related information
        positions = nwbfile.acquisition['position']['player_position'].data[:]
        ptimes = nwbfile.acquisition['position']['player_position'].timestamps[:]
        speed = nwbfile.processing['position_measures']['speed'].data[:]

        # Compute the range of position data, and bin definitions
        area_range = compute_pos_ranges(positions)
        area_range = [area_range[0],32]

        # Get trial timing information
        n_trials = len(nwbfile.trials)
        trial_starts = nwbfile.trials.start_time[:]
        trial_stops = nwbfile.trials.stop_time[:]
        
        # Drop manual trials 
        manual_indx = np.where(nwbfile.trials['drive_type'][:] == 'manual')[0]
        manual_start = trial_starts[manual_indx]
        manual_end = trial_stops[manual_indx]

        move_starts = np.delete(trial_starts, manual_indx, axis=0)
        move_stops= np.delete(trial_stops, manual_indx, axis=0)

        # Compute bin edges
        edges = compute_bin_edges(positions, BINS['place'])


        ###########################################################################################
        ## ANALYZE UNITS

        # Loop across all units
        for unit_ind in range(len(nwbfile.units)):

            # Initialize output unit file name
            name = nwbfile.session_id + '_U' + str(unit_ind).zfill(2)

            # Check if unit already run
            if UNITS['SKIP_ALREADY_RUN'] and file_in_list(name, output_files):
                print_status(\
                    RUN['VERBOSE'], 'skipping unit (already run): \tU{:02d}'.format(unit_ind), 1)
                continue

            if UNITS['SKIP_FAILED'] and file_in_list(name, failed_files):
                print_status(\
                    RUN['VERBOSE'], 'skipping unit (failed): \tU{:02d}'.format(unit_ind), 1)
                continue

            print_status(RUN['VERBOSE'], 'running unit: \t\t\tU{:02d}'.format(unit_ind), 1)

            try:

                # Collect information of interest
                unit_info = create_unit_info(nwbfile.units[unit_ind])
                print(unit_info)
                # Get the spike times for the current unit & restrict time to task range
                spikes = nwbfile.units.get_unit_spike_times(unit_ind)
                # Note: do we still need the task time range check (?)
                spikes = get_range(spikes,
                                   nwbfile.intervals['trials']['start_time'][0],
                                   nwbfile.intervals['trials']['stop_time'][-1])

                # Initialize results and add unit metadata
                results = {}
                results['uid'] = int(unit_ind)
                results['session_id'] = nwbfile.session_id
                results['subject_id'] = nwbfile.subject.subject_id
                for field in ['n_spikes', 'firing_rate', 'channel']:
                    results[field] = unit_info[field]

                # Get the spiking data for each trial
                all_trials = epoch_spikes_by_range(spikes, trial_starts, trial_stops, reset=True)


                ###################################################################################
                ## PLACE CELL ANALYSIS
                trial_occupancy = compute_trial_occupancy(positions, ptimes, BINS['place'],move_starts, move_stops, area_range,speed,**OCCUPANCY_TRIAL)
            
                # Posthoc Check 
                t_occupancy = np.nan_to_num(trial_occupancy, nan=0.0)
                
                # Compute ocuupancy values 
                occ = np.sum(t_occupancy, axis=0)
                trial_occupancy[:, occ < OCCUPANCY['minimum']] = np.nan 
                
                # Compute trial place bins
                trial_place_bins = compute_trial_place_bins(spikes, positions, ptimes, BINS['place'],
                                    move_starts, move_stops, area_range,
                                    speed,**PLACE, trial_occupancy = trial_occupancy,flatten=True)
                
                # Removing the spatial bin with the highest firing rate on each trial
                HF_indx = np.argmax(trial_place_bins, axis=1)
                for i, index in enumerate(HF_indx):
                    trial_place_bins[i, index] = np.nan
    
                # Zscore
                z_scores = zscore(trial_place_bins, axis=None)
                trial_place_bins[z_scores > EXCLUSION['zscore']] = np.nan
                                 
                # Place binned firing
                s_bins = np.linspace(area_range[0], area_range[1], BINS['place']+1)
                t_place_bins  = np.nan_to_num(trial_place_bins , nan=0.0)
                place_bins = np.mean(t_place_bins, axis=0)
                place_bins_std = np.std(t_place_bins, axis=0)
                n_trials= t_place_bins.shape[0]
                place_sem = place_bins_std / np.sqrt(n_trials)




                if 'INFO' in METHODS['PLACE']:
                    results['place_info'] = compute_spatial_information(place_bins, occ, normalize=False)

                if 'ANOVA' in METHODS['PLACE']:
                    results['place_anova'] = fit_anova_place(create_df_place(trial_place_bins))
                    
                    
                # Get spike trial positions 
                sspikes =  threshold_spikes_by_values(spikes, ptimes, speed, OCCUPANCY['min_speed'])
                move_spikes = epoch_spikes_by_range(sspikes, move_starts, move_stops)
                spike_pos = []
                for t_spikes in move_spikes:
                    spike_p,indx = get_values_by_times(ptimes, positions, t_spikes,time_threshold = PLACE['time_threshold'])
                    spike_pos.append(spike_p)

###################################################################################
                ## SURROGATE ANALYSIS

                # Create shuffled time series for comparison
                shuffles = shuffle_spikes(spikes,
                                          SURROGATES['approach'],
                                          SURROGATES['n_shuffles'])

                # Collect list of which surrogate analyses are being run & initialize outputs
                surr_analyses = create_methods_list(METHODS)
                surrs = {analysis : \
                    np.zeros(SURROGATES['n_shuffles']) for analysis in surr_analyses}
                print(surr_analyses)
                for ind, shuffle in enumerate(shuffles):
                    surr_trial_place_bins = compute_trial_place_bins(shuffle, positions, ptimes, BINS['place'],
                                             move_starts,move_stops, area_range,
                                             speed, **PLACE, trial_occupancy = trial_occupancy,flatten=True)
    
                    ## Drop Highest Index 
                    HF_indx = np.argmax(surr_trial_place_bins, axis=1)
                    for i, index in enumerate(HF_indx):
                        surr_trial_place_bins[i, index] = np.nan

                    z_scores = zscore(surr_trial_place_bins, axis=None)
                    outlier_threshold = EXCLUSION['zscore']
                    surr_trial_place_bins[z_scores > outlier_threshold] = 0

                    surr_t_place_bins  = np.nan_to_num(surr_trial_place_bins , nan=0.0)
                    surr_place_bins= np.mean(surr_t_place_bins,axis = 0)

                    ## PLACE
                    if 'INFO' in METHODS['PLACE']:
                        surrs['place_info'][ind] = compute_spatial_information(surr_place_bins, occ, normalize=False)


                    if 'ANOVA' in METHODS['PLACE']:
                        surrs['place_anova'][ind] = fit_anova_place(create_df_place(surr_trial_place_bins))
   




                for analysis in surr_analyses:
                    print(analysis)
                    results[analysis + '_surr_p_val'], results[analysis + '_surr_z_score'] = \
                        compute_surrogate_stats(results[analysis], surrs[analysis])

                # Save out unit results
                save_json(results, name + '.json', folder=results_folder)
                
#                 ###################################################################################
#                 ## MAKE REPORT

                # Initialize figure
                grid = make_grid(3, 7, wspace=.5, hspace=.5, figsize=(25, 8),
                     height_ratios=[1.2,.8, 1.2],title='Unit Report: {}-U{}'.format(nwbfile.session_id, unit_ind))

                plt.rcParams.update({'font.size': 15})
                SI = results['place_info']
                F = results['place_anova']
                ax = get_grid_subplot(grid, 0, slice(0,2))
                plot_rasters(spike_pos,ax = ax, vline=None, figsize=(10, 5),color = 'red',alpha =.3, show_axis=True, title='Forward Trials')
                ax.set_xticklabels([])
                ax.set_xlabel('')
                ax.get_xaxis().set_visible(False)
                drop_spines(['top','right','bottom'],ax = ax)

                ax = get_grid_subplot(grid, slice(1,3), slice(0,2))
                ax.plot(s_bins[:-1], place_bins, color = 'red', label='Mean Value')
                ax.fill_between(s_bins[:-1], place_bins - place_sem, place_bins+ place_sem, color = 'red', alpha=0.3)
                ax.set_xlabel('Position on Track (cm)')
                ax.set_ylabel('Firing Rate')
                ax.set_title(f' SI: {np.round(SI,2)}   F: {np.round(F,2)}')

                drop_spines(['top','right'],ax = ax)

                ax = get_grid_subplot(grid, 0, 2)
                P = results['place_info_surr_p_val']
                plot_surrogates(surrs['place_info'], data_value=SI, p_value=None, title=f'P = {P}',
                                                     title_color=color_pvalue(P),ax = ax,alpha = .6,color = 'grey')
                add_vlines(SI , ax, color='darkred', linestyle='solid', linewidth=4)
                ax.plot(SI , 0, 'o', zorder=10, clip_on=False, color='darkred', markersize=10)
                ax.set_xlabel('Spike Info')
                ax.set_ylabel('count')
                drop_spines(['top', 'right'],ax)

                ax = get_grid_subplot(grid, 2, 2)
                P = results['place_anova_surr_p_val']
                plot_surrogates(surrs['place_anova'], data_value=F, p_value=None ,title=f'P = {P}',
                                                     title_color=color_pvalue(P),ax = ax,alpha = .6,color = 'grey')
                add_vlines(F , ax, color='darkred', linestyle='solid', linewidth=4)
                ax.plot(F , 0, 'o', zorder=10, clip_on=False, color='darkred', markersize=10)
                ax.set_xlabel('F - Statistics')
                ax.set_ylabel('count')
                drop_spines(['top', 'right'],ax)

                ax = get_grid_subplot(grid, slice(0,3), slice(3,5))
                plot_heatmap(trial_place_bins, cbar=True,title= create_heatmap_title('Place bins', trial_place_bins), ax=ax)
                
                 # 01: unit information
                plot_text(create_unit_str(unit_info), title='Unit Information',
                          ax=get_grid_subplot(grid, 0, 6))

                # 02: inter-spike intervals
                plot_isis(compute_isis(spikes), bins=100, range=(0, 1.0),
                          ax=get_grid_subplot(grid, 1, 6))

                save_figure(name + '.pdf', reports_folder, close=True)

            except Exception as excp:

                catch_error(UNITS['CONTINUE_ON_FAIL'], name, results_folder / 'zFailed',
                                RUN['VERBOSE'], 'issue running unit #: \t{}')

        # Close NWB file
        io.close()

    print_status(RUN['VERBOSE'], '\n\nCOMPLETED UNIT ANALYSES\n\n', 0)


if __name__ == '__main__':
    main()