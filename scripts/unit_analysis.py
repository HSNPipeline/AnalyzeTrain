"""Analysis script: Train unit analysis."""

import warnings
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt

from convnwb.io import save_json
from convnwb.io.utils import get_files
from convnwb.utils.log import print_status
from convnwb.utils.run import catch_error


from spiketools.spatial.information import compute_spatial_information

from spiketools.stats.permutations import compute_surrogate_stats

from spiketools.plts.spatial import plot_heatmap, create_heatmap_title
from spiketools.plts.stats import plot_surrogates
from spiketools.plts.annotate import color_pvalue
from spiketools.plts.utils import make_grid, get_grid_subplot, save_figure
from spiketools.plts.style import drop_spines

from spiketools.utils.run import create_methods_list
from spiketools.plts.annotate import add_vlines
# Import settings from local file
from settings import (RUN, PATHS, UNITS, METHODS,
                      SURROGATES)

# Import local code functions
import sys

sys.path.append('../code')
from models import create_df_place,  fit_anova_place
from utils import compute_trial_firing_rates, compute_t_occupancy, circular_shuffle_unit_fr
sys.path.append('../scripts')
from settings import RUN, PATHS,UNITS,METHODS, SURROGATES

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
   
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(reports_folder, exist_ok=True)
    os.makedirs(results_folder / 'zFailed', exist_ok=True)
  

    epochSize = 0.1
    numBins = 40
    kernelSize = 8
    numBinsPos = numBins
    
    data_files = get_files(PATHS['DATA'], select='mat')
   

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

        time_offset = behavioral_data['timesoffset']
        right_edges = time_offset + 99

        mstime = behavioral_data['timesoffset']

  
        events_fr = data['events']['fr'][0][0]

        positions = behavioral_data['position']
        positions = (positions + 34) / 68  
        positions[positions < 0] = 0  
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
       
        #occ = np.nanmean(trial_occupancy,axis = 0)

#         ###########################################################################################
#         ## ANALYZE UNITS

        # Loop across all units
        for unit_ind in range(len(events_fr[1])):
            print('unit_ind',unit_ind)

            try:
                spike_name = data['events']['spikeData'][0][0]['spikeNames'][0][0][0][unit_ind]

                elec_labels = pd.DataFrame(data['events']['spikeData'][0][0]['elecLabels'][0][0][0])
                elec_labels['ElectrodeID'] = elec_labels['ElectrodeID'].str[0].astype(int)
                elec_labels['Label'] = elec_labels['Label'].str[0].astype(str)
                elec_labels = elec_labels[elec_labels['Label'] != 'ainp1']

                # Find matching electírode label
                label = None
                for idx, elec_id in enumerate(elec_labels['ElectrodeID'].values):
                    if str(elec_id) in str(spike_name):
                        label = elec_labels['Label'].iloc[idx]
                        break
                units_fr = events_fr[:,unit_ind]

                spike_times = data['events']['spikeData'][0][0]['spikeTimes'][0][0][0][unit_ind]


                # Initialize dictionaries to store spikes in each bin and their indices
                spikes_in_bin = []
                for k in range(len(mstime)):
                    # Find spikes that fall within the current time bin
                    mask = (spike_times >= mstime[k]) & (spike_times <= right_edges[k])
                    spike_indices = np.where(mask)[0]
                    if len(spike_indices) > 0:
                        #spikes_in_bin.append(spike_index[spike_indices])
                        spikes_in_bin.append(spike_indices)
                    else:
                        spikes_in_bin.append(0)
                spike_position = np.zeros(len(positions))


                for k in range(len(spikes_in_bin)):
                    if isinstance(spikes_in_bin[k], np.ndarray) and spikes_in_bin[k].size > 0:
                        spike_position[k] = positions[k]
                    else:
                        spike_position[k] = 0

                spike_position[spike_position < .01] = np.nan



              
                # Plot spike times
                

                

                # Calculate firing rates with additional checks
                trial_place_bins,trial_fr = compute_trial_firing_rates(trial_bin, pos_bin, units_fr, edges_trial, edges_pos, trial_occupancy, kernelSize, epochSize)
                place_bins = np.nanmean(trial_place_bins,axis = 0)
                place_sem = np.nanstd(trial_place_bins,axis = 0)/np.sqrt(trial_place_bins.shape[0])

                fr_bins = np.nanmean(trial_fr,axis = 0)
                fr_sem = np.nanstd(trial_fr,axis = 0)/np.sqrt(trial_fr.shape[0])

                s_bins = np.linspace(0, 40, numBinsPos+1)

                trialNum = behavioral_data['trialNum']
# Find where trial numbers change
                trial_changes = np.where(np.diff(np.concatenate(([1], trialNum.values))))[0]

                results = {}
                results['unit_ind'] = unit_ind
                results['session_id'] = filename
                results['label'] = str(label) if label is not None else None
                results['spike_name'] = str(spike_name) if spike_name is not None else None
                
                results['place_bins'] = place_bins.tolist()
                results['spike_position'] = spike_position.tolist()
                results['trial_changes'] = trial_changes.tolist()   
                results['trial_Num'] = trialNum.tolist()
                results['trial_place_bins'] = trial_place_bins.tolist()
                results['place_sem'] = place_sem.tolist()


                results['trial_fr'] = trial_fr.tolist()
                results['fr_bins'] = fr_bins.tolist()
                results['fr_sem'] = fr_sem.tolist()

                results['s_bins'] = s_bins.tolist()

                results['fr_bins'] = fr_bins.tolist()
                results['fr_sem'] = fr_sem.tolist()

                results['place_info'] = compute_spatial_information(place_bins[:-3], occ[:-3], normalize=False)
                results['place_fr_info'] = compute_spatial_information(fr_bins[:-3], occ[:-3], normalize=False)
                #print(results['place_info'])

                # Create the dataframe
                df = create_df_place(trial_place_bins[:,:-3])
                results['place_anova']= fit_anova_place(df)

                df_fr = create_df_place(trial_fr[:,:-3])
                results['place_fr_anova']= fit_anova_place(df_fr)
                # Check the computed place F-value
                #print('The ANOVA place F-value is {:4.2f}'.format(results['place_anova']))
                shuffles = circular_shuffle_unit_fr(units_fr, SURROGATES['n_shuffles'])
                
                surr_analyses = create_methods_list(METHODS)
                surrs = {analysis : \
                                    np.zeros(SURROGATES['n_shuffles']) for analysis in surr_analyses}
                
                for ind, shuffle in enumerate(shuffles):
                    #surr_place_bins = compute_firing_rates(shuffle, pos_bin, occ, g, numBins, epochSize)
                    surr_trial_place_bins, surr_trial_fr = compute_trial_firing_rates(trial_bin, pos_bin, shuffle, edges_trial, edges_pos, trial_occupancy, kernelSize, epochSize)
                    surr_place_bins= np.nanmean(surr_trial_place_bins,axis = 0)
                    surrs['place_info'][ind] = compute_spatial_information(surr_place_bins[:-3], occ[:-3], normalize=False)
                    surrs['place_anova'][ind] = fit_anova_place(create_df_place(surr_trial_place_bins[:,:-3]))
                    #surr_trial_fr = compute_trial_firing_rates(trial_bin, pos_bin, shuffle, edges_trial, edges_pos, trial_occupancy, kernelSize, epochSize)
                    surr_fr_bins = np.nanmean(surr_trial_fr,axis = 0)
                    surrs['place_fr_info'][ind] = compute_spatial_information(surr_fr_bins[:-3], occ[:-3], normalize=False)
                    surrs['place_fr_anova'][ind] = fit_anova_place(create_df_place(surr_trial_fr[:,:-3]))
                
                for analysis in surr_analyses:
                    results[analysis + '_surr_p_val'], results[analysis + '_surr_z_score'] = \
                            compute_surrogate_stats(results[analysis], surrs[analysis],title = analysis)
                #print(results)
                save_json(results, filename+'_U'+str(unit_ind).zfill(2) + '.json', folder=results_folder)

                plt.rcParams.update({'font.size': 20})
                grid = make_grid(6, 7, wspace=.8, hspace=1, figsize=(30, 20),
                                    title = filename+'_U'+str(unit_ind).zfill(2))

                #plt.rcParams.update({'font.size': 25})
                SI = results['place_info']
                F = results['place_anova']


                ax = get_grid_subplot(grid, slice(1,3), slice(0,2))
                ax.plot( s_bins[:-2], place_bins[:-1], color = 'red', label='Mean Value')
                ax.fill_between( s_bins[:-2], place_bins[:-1] - place_sem[:-1], place_bins[:-1]+ place_sem[:-1], color = 'red', alpha=0.1)
                ax.set_xlabel('Position on Virtual Track (cm)')
                ax.set_ylabel('Firing Rate')
                ax.set_title(f' SI: {np.round(SI,2)}   F: {np.round(F,2)}')
                drop_spines(['top','right'],ax = ax)

                spikes_positions_trials = [[] for _ in range(64)]

                # Create a figure
                ax = get_grid_subplot(grid, slice(0,1), slice(0,2))

                # For each trial

                for num_trial in range(64):
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
                                    'k.', markersize=3,alpha = .5)
                            
                drop_spines(['top','right','bottom'],ax = ax)
                ax.set_xticklabels([])
                ax.set_xticks([])





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



                ax = get_grid_subplot(grid, slice(0,3), slice(3,6))
                plot_heatmap(trial_place_bins, cbar=True,title= create_heatmap_title('Place bins', trial_place_bins), ax=ax)


                name = filename+'_U'+str(unit_ind).zfill(2)
                save_figure(name + '.pdf', reports_folder, close=True)

            except Exception as excp:

                catch_error(UNITS['CONTINUE_ON_FAIL'], name, results_folder / 'zFailed',
                                RUN['VERBOSE'], 'issue running unit #: \t{}')



    print_status(RUN['VERBOSE'], '\n\nCOMPLETED UNIT ANALYSES\n\n', 0)


if __name__ == '__main__':
    main()
