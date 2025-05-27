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
from utils import compute_trial_place_bins, compute_t_occupancy, circular_shuffle_unit_fr
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
    results_folder = PATHS['RESULTS'] / 'units_bins'
    reports_folder = PATHS['REPORTS'] / 'units_bins'
   
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(reports_folder, exist_ok=True)
    os.makedirs(results_folder / 'zFailed', exist_ok=True)
  
    bin_sets = [20,30,40,50,60]
    epochSize = 0.1
    

    
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

  
        events_fr = data['events']['fr'][0][0]

        positions = behavioral_data['position']
        positions = (positions + 34) / 68  
        positions[positions < 0] = 0  
        positions[positions > 1] = 1 


        #occ = np.nanmean(trial_occupancy,axis = 0)

#         ###########################################################################################
#         ## ANALYZE UNITS

        # Loop across all units
        for unit_ind in range(len(events_fr[1])):
            print('unit_ind',unit_ind)
            results = {}
            results['unit_ind'] = unit_ind
            results['session_id'] = filename
            
            try:    
                for numBins in bin_sets:

                    results['numBins'] = numBins
                    edges_pos = np.linspace(0, 100, numBins+1)
                    pos_bin = np.digitize(positions*100, edges_pos)

                    count, _ = np.histogram(positions*100, bins=edges_pos)
                    
                    counts = count[:numBins]
                    occ = counts * epochSize
            
                        

                    # Bin by trial number
                    edges_trial = np.arange(0.5, 65, 1)  # 0.5:1:65 in MATLAB
                    trial_bin = np.array(behavioral_data['trialNum'])

                    trial_occupancy = compute_t_occupancy(trial_bin, pos_bin, edges_trial, edges_pos, epochSize)
                    units_fr = events_fr[:,unit_ind]

                    trial_place_bins= compute_trial_place_bins(trial_bin, pos_bin, units_fr, edges_trial, edges_pos, trial_occupancy, epochSize)
                    place_bins = np.nanmean(trial_place_bins,axis = 0)
                    
                    results[f'place_bins_{numBins}'] = place_bins.tolist()
                    
                    results[f'trial_place_bins_{numBins}'] = trial_place_bins.tolist()
                    
                    results[f'place_info_{numBins}'] = compute_spatial_information(place_bins[:-3], occ[:-3], normalize=False)
                  
                    df = create_df_place(trial_place_bins[:,:-3])
                    results[f'place_anova_{numBins}']= fit_anova_place(df)

                save_json(results, filename+'_U'+str(unit_ind).zfill(2) +  '.json', folder=results_folder)

            except Exception as excp:

                catch_error(UNITS['CONTINUE_ON_FAIL'], filename, results_folder / 'zFailed',
                                RUN['VERBOSE'], 'issue running unit #: \t{}')


    print_status(RUN['VERBOSE'], '\n\nCOMPLETED UNIT ANALYSES\n\n', 0)


if __name__ == '__main__':
    main()
