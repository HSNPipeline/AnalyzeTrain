"""Run analyses across sessions."""

import numpy as np

import matplotlib.pyplot as plt

from convnwb.io import load_nwbfile,get_files
from convnwb.utils.log import print_status
from convnwb.io.utils import get_files, file_in_list

from spiketools.measures.spikes import compute_firing_rate
from spiketools.plts.spatial import plot_position_by_time, plot_position_1d, plot_heatmap,create_heatmap_title
from spiketools.plts.spikes import plot_firing_rates


from spiketools.plts.data import  plot_hist, plot_text,plot_lines
from spiketools.plts.annotate import add_hlines, add_vlines

from spiketools.utils.timestamps import convert_sec_to_min
from spiketools.plts.utils import make_grid, get_grid_subplot

from spiketools.utils.timestamps import convert_sec_to_min
from spiketools.spatial.occupancy import compute_occupancy

# Import settings from local file
from settings import RUN, PATHS

# Local imports
import sys
sys.path.append('../code')
from plts import plot_positions_with_speed
from utils import group_array_by_key
from reports import create_sess_str

###################################################################################################
###################################################################################################

def main():
    """Run session analyses."""

    base_path = "/Users/weijiazhang/Data/Train/nwbfiles"
    nwbfiles = get_files(base_path, select='nwb')
    print(nwbfiles)
    
    
    # Define output folders
    report_path = '/Users/weijiazhang/Data/Train/report/session_report/'

    for nwbfilename in nwbfiles:

        ## LOADING & DATA ACCESSING

        # Check and ignore files set to ignore
        if file_in_list(nwbfilename, RUN['IGNORE']):
            print_status(RUN['VERBOSE'], 'Skipping file (set to ignore): {}'.format(nwbfilename), 0)
            continue

        # Load file and prepare data
        print_status(RUN['VERBOSE'], 'Running session analysis: {}'.format(nwbfilename), 0)

        # Load NWB file
        nwbfile, io = load_nwbfile(nwbfilename, base_path, return_io=True)
      
        ## EXTRACT DATA OF INTEREST
        print('')

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
        
        # Check for original object positions 
        obj_pos = nwbfile.trials['object_position'].data[:]
        obj = nwbfile.trials['object'].data[:]
        obj_res = group_array_by_key(obj, obj_pos)
        obj_barrel = np.array(obj_res ['barrel'][:])
        obj_box = np.array(obj_res ['box'][:])
        obj_desk = np.array(obj_res ['desk'][:])
        obj_bench = np.array(obj_res ['bench'][:])

        # Check for subjects' recall object positions
        res_pos = nwbfile.trials['response_position'].data[:]
        result_res = group_array_by_key(obj, res_pos)
        res_barrel = np.array(result_res ['barrel'][:])
        res_box = np.array(result_res ['box'][:])
        res_desk = np.array(result_res ['desk'][:])
        res_bench = np.array(result_res ['bench'][:])
        
        
        # Check for available units 
        n_units = len(nwbfile.units)
        print('Number of unit: {}'.format(n_units))
        summary['n_units'] = n_units
        
        # Get spiking activity from across all units
        all_spikes = [nwbfile.units.get_unit_spike_times(uind) for uind in range(n_units)]

        # Calculate the average overall firing rate of each neuron
        rates = [compute_firing_rate(spikes) for spikes in all_spikes] 


    
    
        ## PRECOMPUTE MEASURES OF INTEREST
        ...

        ## CREATE SESSION REPORT
        grid = make_grid(5, 5, wspace=0.4, hspace=0.5, figsize=(15, 25),
                     width_ratios=[0.7, 0.7, 0.7, 0.7, 0.7],
                     title=f'Session Report: {nwbfilename}')
        plot_text(create_sess_str(summary), ax=get_grid_subplot(grid, 0, 0))
        plot_heatmap(occ, title=create_heatmap_title('Occupancy-',occ), ax=get_grid_subplot(grid, slice(1,2), slice(0, 2)))
        plot_position_by_time(ptimes,positions,ax=get_grid_subplot(grid, slice(0, 1), slice(2, 5)))
        plt.title('Positions')

        plot_firing_rates(rates, ax = get_grid_subplot(grid, slice(1, 2), slice(2, 5)))

        plot_position_1d(positions, [obj_barrel,obj_box,obj_desk,obj_bench],ax = get_grid_subplot(grid, slice(2, 3), slice(2, 5)))
        plt.title('Object positions')

        plot_position_1d(positions, [res_barrel, res_box, res_desk, res_bench],
                         title='Response Positions', ax = get_grid_subplot(grid, slice(3, 4), slice(2, 5)))

        plot_hist(speed, bins=25,ax=get_grid_subplot(grid, 0,1))
        add_vlines(speed_thresh, color='red')
        plt.title('Speed')

        plot_lines(ptimes, speed, marker='.', title='Speed - Whole Task',ax=get_grid_subplot(grid, slice(2,3), slice(0, 2)))
        add_hlines(speed_thresh, color='red', alpha=0.75, linestyle='--')
        plot_positions_with_speed(ptimes, positions, speed,speed_thresh, ax=get_grid_subplot(grid, slice(3,4), slice(0, 2)))

        plt.savefig(f'/Users/weijiazhang/Data/Train/report/session_report/{nwbfilename}_report.pdf')



        # Close the nwbfile
        io.close()

    print_status(RUN['VERBOSE'], '\n\nCOMPLETED SESSION ANALYSES\n\n', 0)

if __name__ == '__main__':
    main()
