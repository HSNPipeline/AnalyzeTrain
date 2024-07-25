"""Analysis script: T3 unit analysis."""

import warnings

import numpy as np

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
from spiketools.plts.annotate import color_pvalue
from spiketools.plts.utils import make_grid, get_grid_subplot, save_figure
from spiketools.plts.style import drop_spines
from spiketools.utils.extract import (get_range, get_values_by_time_range,
                                      get_values_by_times, threshold_spikes_by_values)
from spiketools.utils.epoch import epoch_spikes_by_range, epoch_spikes_by_event
from spiketools.utils.trials import split_trials_by_condition
from spiketools.utils.base import add_key_prefix, combine_dicts
from spiketools.utils.run import create_methods_list

# Import settings from local file
from settings import (RUN, PATHS, UNITS, METHODS, BINS, OCCUPANCY,
                      PLACE,  SURROGATES, RUN_SPECIFIC)

# Import local code functions
import sys
sys.path.append('../code')
from models import (#create_df_chest, fit_anova_chest,
                    create_df_place, fit_anova_place,)
                    #create_df_splitter_place, fit_anova_splitter_place,
                    #create_df_time, fit_anova_time,
                    #create_df_splitter_time, fit_anova_splitter_time)
    
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
    results_folder = PATHS['RESULTS'] / 'units'
    reports_folder = PATHS['REPORTS'] / 'units'
    print(reports_folder)
    # Collect a copy of all settings with a prefixes
    all_settings = [
        add_key_prefix(BINS, 'bins'),
        add_key_prefix(OCCUPANCY, 'occupancy'),
        add_key_prefix(PLACE, 'place'),
        #add_key_prefix(TIMES, 'times'),
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
        
#         # Get chest locations
#         chests = {
#             'left' : nwbfile.stimulus['left_chest_position'].data[:],
#             'right' : nwbfile.stimulus['right_chest_position'].data[:],
#         }

        # Get position related information
        positions = nwbfile.acquisition['position']['player_position'].data[:]
        ptimes = nwbfile.acquisition['position']['player_position'].timestamps[:]
        speed = nwbfile.processing['position_measures']['speed'].data[:]

        # Compute the range of position data, and bin definitions
        area_range = compute_pos_ranges(positions)
        area_range = [-32,32]
#         splitter_bins = compute_bin_edges(None, BINS['nbins_stem'], BINS['stem_bin_range'])

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


#                 ###################################################################################
#                 ## PLACE CELL ANALYSIS

                # Compute occupancy values
                occ = compute_occupancy(positions, ptimes, BINS['place'],area_range,
                                        speed=speed, **OCCUPANCY)
                trial_occupancy = compute_trial_occupancy(positions, ptimes, BINS['place'],move_starts, move_stops, area_range,speed,**OCCUPANCY)

                ## Place binned firing
                place_bins = compute_place_bins(spikes, positions, ptimes, BINS['place'],
                                                area_range, speed, **PLACE, occupancy=occ)

                if 'INFO' in METHODS['PLACE']:
                    results['place_info'] = compute_spatial_information(\
                        place_bins, occ, normalize=False)

                if 'ANOVA' in METHODS['PLACE']:
                    trial_fr = compute_trial_place_bins(\
                        spikes, positions, ptimes, BINS['place'], move_starts, move_stops,area_range, speed, **PLACE,trial_occupancy = trial_occupancy, flatten=True)
                    results['place_anova'] = fit_anova_place(create_df_place(trial_fr))
                    
                    
                # Get spike trial positions 
                sspikes =  threshold_spikes_by_values(spikes, ptimes, speed, OCCUPANCY['min_speed'])
                move_spikes = epoch_spikes_by_range(sspikes, move_starts, move_stops)
                spike_pos = []
                for t_spikes in move_spikes:
                    spike_p,indx = get_values_by_times(ptimes, positions, t_spikes,time_threshold = PLACE['time_threshold'])
                    spike_pos.append(spike_p)
                
                # PLOTS 
                s_bins = np.linspace(area_range[0], area_range[1], BINS['place']+1)
                Place_bins = np.mean(trial_fr, axis=0)
                place_std = np.std(trial_fr, axis=0)
                place_n= trial_fr.shape[0]
                place_sem = place_std  / np.sqrt(place_n)


#                 ###################################################################################
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
                  

                    ## PLACE
                    if 'INFO' in METHODS['PLACE']:
                        surr_place_bins = compute_place_bins(\
                            shuffle, positions, ptimes, BINS['place'], area_range,
                            speed, **PLACE, occupancy=occ)
                        surrs['place_info'][ind] = compute_spatial_information(\
                            surr_place_bins, occ, normalize=False)

                    if 'ANOVA' in METHODS['PLACE']:
                        surr_place_trial = compute_trial_place_bins(\
                            shuffle, positions, ptimes, BINS['place'],
                            move_starts, move_stops, area_range,
                            speed, **PLACE, trial_occupancy = trial_occupancy, flatten=True)
                        surrs['place_anova'][ind] = \
                            fit_anova_place(create_df_place(surr_place_trial))



#                 # Compute surrogate statistics
                for analysis in surr_analyses:
                    print(analysis)
                    results[analysis + '_surr_p_val'], results[analysis + '_surr_z_score'] = \
                        compute_surrogate_stats(results[analysis], surrs[analysis])

                # Save out unit results
                save_json(results, name + '.json', folder=results_folder)
                
#                 ###################################################################################
#                 ## MAKE REPORT

                # Initialize figure
                grid = make_grid(8, 3, wspace=0.4, hspace=0.65,width_ratios=[1.5,1,1], figsize=(15, 25),
                                 title='Unit Report: {}-U{}'.format(nwbfile.session_id, unit_ind))

#                 # 00: spike waveform
#                 plot_waveform(np.array(nwbfile.units['waveforms'][unit_ind][:]),
#                               average='mean', shade='std',
#                               ax=get_grid_subplot(grid, 0, 0))

                # 01: unit information
                plot_text(create_unit_str(unit_info), title='Unit Information',
                          ax=get_grid_subplot(grid, 0, 1))

                # 02: inter-spike intervals
                plot_isis(compute_isis(spikes), bins=100, range=(0, 1.0),
                          ax=get_grid_subplot(grid, 0, 2))

                # 10: raster across all trials
                # TODO: replace with everything raster
                plot_rasters(all_trials, title='All Trials',
                             ax=get_grid_subplot(grid, 1, slice(0, 2)))


                # 40: place bin firing
                plot_heatmap(place_bins, title=create_heatmap_title('Place Bins', place_bins),
                             ax=get_grid_subplot(grid, 3, slice(0,2)))

#                 # 41: splitter place activity
#                 splitter_plot(positions, stem_spikes_sides, splitter_bins, BINS['stem_bin_range'],
#                               ax=get_grid_subplot(grid, slice(4, 6), 1))

                # 42: place surrogates (info)
                if 'INFO' in METHODS['PLACE']:
                    plot_surrogates(surrs['place_info'],
                                    results['place_info'],
                                    results['place_info_surr_p_val'],
                                    title='Place Surrogates (INFO)',
                                    title_color=color_pvalue(results['place_info_surr_p_val']),
                                    ax=get_grid_subplot(grid, 4, 1))

                # 42: place surrogates (anova)
                if 'ANOVA' in METHODS['PLACE']:
                    plot_surrogates(surrs['place_anova'],
                                    results['place_anova'],
                                    results['place_anova_surr_p_val'],
                                    title='Place Surrogates (ANOVA)',
                             title_color=color_pvalue(results['place_anova_surr_p_val']),
                                    ax=get_grid_subplot(grid, 5, 1))
                                   
                                    
                ax = get_grid_subplot(grid, 4, 0)
                plot_rasters(spike_pos,ax =ax, vline=None,color = 'red',alpha =.4,show_axis=True, title=None)
                ax.set_ylabel('Trials')
                drop_spines(['top','bottom','right'],ax)
                ax.set_xticklabels([])
                ax.set_xlabel('')
                ax.get_xaxis().set_visible(False)
                #ax.set_title(create_heatmap_title(f'U{uid}-FR: ', Place_bins, stat=z_score, p_val=F_p_val), color=color_pvalue(p_val, alpha=0.05))

                #add_vlines([Init_end,Inter_end], ax=ax, color='grey', lw=3,linestyle='--', alpha=0.5)

                ax = get_grid_subplot(grid, slice(5,7), 0)
                ax.plot(s_bins[:-1], Place_bins, linewidth = '2',color = 'red', label='seqA')
                ax.fill_between(s_bins[:-1],Place_bins - place_sem, Place_bins + place_sem, color = 'red', alpha=0.3)
                ax.set_xlabel('Position on Track')
                ax.set_ylabel('Firing Rate')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                SI = results['place_info']
                F = results['place_anova']
                ax.set_title(f'SI: {np.round(SI,2)} F: {np.round(F,2)}')
                ax.set_xlabel('')
                #add_vlines([Init_end,Inter_end], ax=ax, color='grey', lw=3,linestyle='--', alpha=0.5)

#                 ## 05 seqA Learning: SI Surrogate analysis
#                 ax = get_grid_subplot(grid, 5, 1)
#                 plot_surrogates(info_surrs, data_value=place_info, p_value=SI_p_val,ax = ax,alpha = .6,color = 'grey')
#                 add_vlines(spike_info, ax=ax, color='darkred', linestyle='solid', linewidth=4)
#                 ax.plot(spike_info, 0, 'o', zorder=10, clip_on=False, color='darkred', markersize=10)
#                 ax.set_xlabel('Spike Info')
#                 ax.set_ylabel('count')
#                 drop_spines(['top', 'right'],ax)


#                 ## 06 seqA Learning: ANOVA Surrogate analsys
#                 ax = get_grid_subplot(grid, 5, 1)
#                 plot_surrogates(anova_surrs, data_value=f_val, p_value=F_p_val,ax = ax,alpha = .6,color = 'grey')
#                 add_vlines(f_val, ax=ax, color='darkred', linestyle='solid', linewidth=4)
#                 ax.plot(f_val, 0, 'o', zorder=10, clip_on=False, color='darkred', markersize=10)
#                 ax.set_xlabel('ANOVA')
#                 ax.set_ylabel('count')
#                 drop_spines(['top', 'right'],ax)

                # Plot the occupancy
                ax = get_grid_subplot(grid,2,slice(0,2))
                plot_heatmap(occ, title= create_heatmap_title('Occupancy', occ),ax = ax)


                # Plot the Trial occupancy
                ax = get_grid_subplot(grid,slice(1,3),slice(2,3))
                plot_heatmap(trial_occupancy, title= create_heatmap_title('Trial Occupancy', trial_occupancy),ax = ax)
                
                ax = get_grid_subplot(grid,slice(4,6),slice(2,3))
                plot_heatmap(trial_fr, title= create_heatmap_title('Trial Firing Rate', trial_fr),ax = ax)
                 # Save out the report
                save_figure(name + '.pdf', reports_folder, close=True)

            except Exception as excp:

                catch_error(UNITS['CONTINUE_ON_FAIL'], name, results_folder / 'zFailed',
                            RUN['VERBOSE'], 'issue running unit #: \t{}')

        # Close NWB file
        io.close()

    print_status(RUN['VERBOSE'], '\n\nCOMPLETED UNIT ANALYSES\n\n', 0)


if __name__ == '__main__':
    main()
