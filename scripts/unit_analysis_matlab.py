"""Analysis script: Train unit analysis."""

import warnings
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt

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
from spiketools.plts.annotate import add_vlines
# Import settings from local file
from settings import (RUN, PATHS, UNITS, METHODS, BINS, OCCUPANCY,
                      PLACE,  SURROGATES, RUN_SPECIFIC)

# Import local code functions
import sys

sys.path.append('../code')
from models import PLACE_MODELS, PLACE_COND_MODELS, create_df_place, create_df_place_cond, fit_anova_place, fit_anova_place_cond
from utils import compute_firing_rates, compute_trial_firing_rates, compute_t_occupancy, circular_shuffle_unit_fr
sys.path.append('../scripts')
from settings import RUN, PATHS,OCCUPANCY,OCCUPANCY_TRIAL, PLACE,UNITS,METHODS, SURROGATES,BINS

#from plts import COLORS, splitter_plot

###################################################################################################
###################################################################################################

def main():
    """Run analyses across all units."""

    # Supress ANOVA warnings. ToDo: fix this later
    warnings.filterwarnings("ignore")

    print_status(RUN['VERBOSE'], '\n\nANALYZING UNIT DATA - {}\n\n'.format(RUN['TASK']), 0)

    # Define output folders
    results_folder = PATHS['RESULTS'] / 'units_matlab'
    reports_folder = PATHS['REPORTS'] / 'units_matlab'
   
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(reports_folder, exist_ok=True)
    os.makedirs(results_folder / 'zFailed', exist_ok=True)
  

    epochSize = 0.1
    numBins = 40
    kernelSize = 8
    numBinsPos = 40
    
    data_files = get_files(PATHS['DATA'], select='mat')
   

    output_files = get_files(results_folder, select='json', drop_extensions=True)
    failed_files = get_files(results_folder / 'zFailed', select='json', drop_extensions=True)

    for filename in data_files:
        print(filename)

        # Load NWB file
        data = loadmat( PATHS['DATA'] / filename)
       # print(data)
        
        # Get behavioral data
        df = pd.DataFrame(data['events']['events'][0][0][0])
        # Convert object columns to string type
        str_cols = ['subj', 'object','timesfile']
        for col in str_cols:
            df[col] = df[col].str[0].astype(str)
        # Convert numeric columns to appropriate types
        int_cols = ['session', 'trialNum', 'blockType', 'driveType'] 
        float_cols = ['object_position', 'response_position', 'response_mstime', 'timesoffset','mstime', 'position','speed']
        for col in int_cols:
            df[col] = df[col].str[0].astype(int)
        for col in float_cols:
            df[col] = df[col].str[0].astype(float)

        behavioral_data = df

        # Get speed related information
        speeds = np.array(behavioral_data['speed'])
        # Drop bad speed data
        bad = speeds < 2  
        behavioral_data = behavioral_data[~bad]

        events_fr = data['events']['fr'][0][0]
        events_fr = events_fr[~bad,:]


               # Get position related information
        positions = behavioral_data['position']
        print(positions)
        positions = (positions + 34) / 68  # Normalize to 0-1 range
        positions[positions < 0] = 0  # Clip values below 0
        positions[positions > 1] = 1 
        

        edges_pos = np.linspace(0, 100, numBinsPos+1)
        pos_bin = np.digitize(positions*100, edges_pos)


        count, _ = np.histogram(positions*100, bins=edges_pos)
        
        counts = count[:numBins]
        occ = counts * epochSize
        pos_bin.shape
                

        g = signal.windows.gaussian(kernelSize, std=1)
        g = g/np.sum(g)             

        # Bin by trial number
        edges_trial = np.arange(0.5, 65, 1)  # 0.5:1:65 in MATLAB
        trial_bin = np.array(behavioral_data['trialNum'])

        trial_occupancy = compute_t_occupancy(trial_bin, pos_bin, edges_trial, edges_pos, epochSize)
        print(trial_occupancy)
        #occ = np.nanmean(trial_occupancy,axis = 0)

#         ###########################################################################################
#         ## ANALYZE UNITS

        # Loop across all units
        for unit_ind in range(len(events_fr[1])):
            print('unit_ind',unit_ind)
#             # Initialize output unit file name
#             name = nwbfile.session_id + '_U' + str(unit_ind).zfill(2)

#             # Check if unit already run
#             if UNITS['SKIP_ALREADY_RUN'] and file_in_list(name, output_files):
#                 print_status(\
#                     RUN['VERBOSE'], 'skipping unit (already run): \tU{:02d}'.format(unit_ind), 1)
#                 continue

#             if UNITS['SKIP_FAILED'] and file_in_list(name, failed_files):
#                 print_status(\
#                     RUN['VERBOSE'], 'skipping unit (failed): \tU{:02d}'.format(unit_ind), 1)
#                 continue

#             print_status(RUN['VERBOSE'], 'running unit: \t\t\tU{:02d}'.format(unit_ind), 1)

            try:
                units_fr = events_fr[:,unit_ind]

                # Calculate firing rates with additional checks
                trial_place_bins = compute_trial_firing_rates(trial_bin, pos_bin, units_fr, edges_trial, edges_pos, trial_occupancy, kernelSize, epochSize)
                place_bins = np.nanmean(trial_place_bins,axis = 0)
                place_sem = np.nanstd(trial_place_bins,axis = 0)/np.sqrt(trial_place_bins.shape[0])

                s_bins = np.linspace(0, 40, numBinsPos+1)

                results = {}
                results['unit_ind'] = unit_ind
                results['session_id'] = filename
                
                results['place_bins'] = place_bins.tolist()
                results['place_bins_trial'] = trial_place_bins.tolist()
                results['place_sem'] = place_sem.tolist()
                results['s_bins'] = s_bins.tolist()

                results['place_info'] = compute_spatial_information(place_bins[:-3], occ[:-3], normalize=False)
                print(results['place_info'])

                # Create the dataframe
                df = create_df_place(trial_place_bins[:,:-3])
                results['place_anova']= fit_anova_place(df)
                # Check the computed place F-value
                print('The ANOVA place F-value is {:4.2f}'.format(results['place_anova']))
                shuffles = circular_shuffle_unit_fr(units_fr, SURROGATES['n_shuffles'])
                
                surr_analyses = create_methods_list(METHODS)
                surrs = {analysis : \
                                    np.zeros(SURROGATES['n_shuffles']) for analysis in surr_analyses}
                
                for ind, shuffle in enumerate(shuffles):
                    #surr_place_bins = compute_firing_rates(shuffle, pos_bin, occ, g, numBins, epochSize)
                    surr_trial_place_bins = compute_trial_firing_rates(trial_bin, pos_bin, shuffle, edges_trial, edges_pos, trial_occupancy, kernelSize, epochSize)
                    surr_place_bins= np.nanmean(surr_trial_place_bins,axis = 0)
                    surrs['place_info'][ind] = compute_spatial_information(surr_place_bins[:-3], occ[:-3], normalize=False)
                    surrs['place_anova'][ind] = fit_anova_place(create_df_place(surr_trial_place_bins[:,:-3]))
                
                for analysis in surr_analyses:
                    results[analysis + '_surr_p_val'], results[analysis + '_surr_z_score'] = \
                            compute_surrogate_stats(results[analysis], surrs[analysis],title = analysis)

                save_json(results, filename+'_U'+str(unit_ind).zfill(2) + '.json', folder=results_folder)

                plt.rcParams.update({'font.size': 20})
                grid = make_grid(3, 6, wspace=.8, hspace=.8, figsize=(30, 10),
                                    height_ratios=[1,1, 1.2],title = filename+'_U'+str(unit_ind).zfill(2))

                #plt.rcParams.update({'font.size': 25})
                SI = results['place_info']
                F = results['place_anova']
                # ax = get_grid_subplot(grid, 0, slice(0,2))
                # plot_rasters(spike_pos,ax = ax, vline=None, figsize=(10, 5),color = 'red',alpha =.3, show_axis=True, title='Forward Trials')
                # ax.set_xticklabels([])
                # ax.set_xlabel('')
                # ax.get_xaxis().set_visible(False)
                # drop_spines(['top','right','bottom'],ax = ax)

                ax = get_grid_subplot(grid, slice(1,3), slice(0,2))
                ax.plot( s_bins[:-1], place_bins, color = 'red', label='Mean Value')
                ax.fill_between( s_bins[:-1], place_bins - place_sem, place_bins+ place_sem, color = 'red', alpha=0.1)
                add_vlines(37, ax, color='red', linestyle='solid', linewidth=4)
                ax.set_xlabel('Position on Virtual Track (cm)')
                ax.set_ylabel('Firing Rate')
                ax.set_title(f' SI: {np.round(SI,2)}   F: {np.round(F,2)}')
                drop_spines(['top','right'],ax = ax)

                ax = get_grid_subplot(grid, 1, 2)
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

                ax = get_grid_subplot(grid, slice(1,3), slice(3,5))
                plot_heatmap(trial_place_bins, cbar=True,title= create_heatmap_title('Place bins', trial_place_bins), ax=ax)
                name = filename+'_U'+str(unit_ind).zfill(2)
                save_figure(name + '.pdf', reports_folder, close=True)
            #save_figure(grid, PATHS['REPORTS'] / filename / ('U' + str(unit_ind).zfill(2) + '.pdf'))
            #save_json(results, PATHS['RESULTS'] / name / ('U' + str(unit_ind).zfill(2) + '.json'))
            except Exception as excp:

                catch_error(UNITS['CONTINUE_ON_FAIL'], name, results_folder / 'zFailed',
                                RUN['VERBOSE'], 'issue running unit #: \t{}')

        # Close NWB file
        #io.close()

    print_status(RUN['VERBOSE'], '\n\nCOMPLETED UNIT ANALYSES\n\n', 0)


if __name__ == '__main__':
    main()
