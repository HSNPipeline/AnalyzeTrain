"""Helper functions for creating reports."""

import numpy as np

###################################################################################################
###################################################################################################
def create_sess_str(session):
    """Create a string representation of the group info."""

    string = '\n'.join([
        '\n',
        'experiment:  {}'.format(session['experiment']),
        'subject:  {}'.format(session['subject']),
        'session:  {}'.format(session['session']),
        'number of trials: {}'.format(session['n_trials']),
        'task length: {}'.format(session['task_length']),
        'number of units: {}'.format(session['n_units'])
    ])

    return string

"""Helper functions for creating reports."""

import numpy as np

from spiketools.measures.spikes import compute_firing_rate
from spiketools.utils.timestamps import convert_sec_to_min

# ###################################################################################################
# ###################################################################################################

# ## GROUP

# def create_group_info(summary):
#     """Create a dictionary of group information."""

#     group_info = {}
#     group_info['n_subjects'] = len(set([el.split('-')[0] for el in summary['ids']]))
#     group_info['n_sessions'] = len(summary['ids'])

#     return group_info


# def create_group_str(group_info):
#     """Create a string representation of the group information."""

#     string = '\n'.join([
#         'Number of subjects:    {:10d}'.format(group_info['n_subjects']),
#         'Number of sessions:    {:10d}'.format(group_info['n_sessions']),
#     ])

#     return string

# def create_group_sessions_str(summary):
#     """Create strings of detailed session information."""

#     out = []
#     strtemp = "{}: Neural -{:3d} units | Behav - {:3d} trials, {:3.0f}% alternation, avg error: {:5.2f}"
#     for ind in range(len(summary['ids'])):
#         out.append(strtemp.format(summary['ids'][ind],
#                                   summary['n_units'][ind],
#                                   summary['n_trials'][ind],
#                                   summary['alternation'][ind] * 100,
#                                   summary['error'][ind]))

#     return out


# ## SESSION

# def create_subject_info(nwbfile):
#     """Create a dictionary of subject information."""

#     subject_info = {}

#     st = nwbfile.intervals['trials'][0]['start_time'].values[0]
#     en = nwbfile.intervals['trials'][-1]['stop_time'].values[0]

#     subject_info['n_units'] = len(nwbfile.units)
#     subject_info['n_trials'] = len(nwbfile.intervals['trials'])
#     subject_info['subject_id'] = nwbfile.subject.subject_id
#     subject_info['session_id'] = nwbfile.session_id
#     subject_info['trials_start'] = st
#     subject_info['trials_end'] = en
#     subject_info['session_length'] = float(convert_sec_to_min(en))

#     return subject_info


# def create_subject_str(subject_info):
#     """Create a string representation of the subject / session information."""

#     string = '\n'.join([
#         'Recording:  {:5s}'.format(subject_info['session_id']),
#         'Number of units:    {:10d}'.format(subject_info['n_units']),
#         'Number of trials:   {:10d}'.format(subject_info['n_trials']),
#         'Session length:     {:.2f}'.format(subject_info['session_length'])
#     ])

#     return string


# def create_position_str(bins, occ, chests):
#     """Create a string representation of position information."""

#     string = '\n'.join([
#         'Position bins: {:2d}, {:2d}'.format(*bins),
#         'Median occupancy: {:2.4f}'.format(np.nanmedian(occ)),
#         'Min / Max occupancy:  {:2.4f}, {:2.4f}'.format(np.nanmin(occ), np.nanmax(occ)),
#         'Left chest location: {:2.2f}, {:2.2f}'.format(*chests['left']),
#         'Right chest location: {:2.2f}, {:2.2f}'.format(*chests['right']),
#     ])

#     return string


# def create_behav_info(behav):
#     """Create a dictionary of session behaviour information."""

#     behav_info = {}

#     behav_info['n_trials'] = len(behav)
#     behav_info['n_encoding'] = sum(behav['trial_type'] == 'encoding')
#     behav_info['n_retrieval'] = sum(behav['trial_type'] == 'retrieval')
#     behav_info['complete_session'] = len(behav) == 36

#     turn_counts_tot = behav.turn_correctness.value_counts()
#     behav_info['turn_correct'] = turn_counts_tot.get('True', 0)
#     behav_info['turn_total'] = turn_counts_tot.get('True', 0) + turn_counts_tot.get('False', 0)

#     cp_split = behav.groupby('trial_type')['turn_correctness'].value_counts()
#     cp_enc = cp_split.get('encoding')
#     cp_ret = cp_split.get('retrieval')

#     enc_true = cp_enc.get('True', 0) if cp_enc is not None else 0
#     enc_total = cp_enc.sum() if cp_enc is not None else 0
#     ret_true = cp_ret.get('True', 0) if cp_ret is not None else 0
#     ret_total = cp_ret.sum() if cp_ret is not None else 0

#     behav_info['cp_enc_cor'] = enc_true
#     behav_info['cp_enc_tot'] = enc_total
#     behav_info['cp_ret_cor'] = ret_true
#     behav_info['cp_ret_tot'] = ret_total

#     behav_info['cp_enc_perc'] = (enc_true / enc_total) if enc_total > 0 else 0
#     behav_info['cp_ret_perc'] = (ret_true / ret_total) if ret_total > 0 else 0

#     behav_info['n_responses'] = len(behav['button_press_time'].dropna())
#     behav_info['err_all'] = behav.groupby(['turn_correctness'])['error'].mean()['True']

#     err_sides = behav[behav.turn_correctness == 'True'].\
#         groupby(['correct_direction'])['error'].mean()
#     behav_info['err_left'] = err_sides['left']
#     behav_info['err_right'] = err_sides['right']

#     return behav_info


# def create_behav_str(behav_info):
#     """Create a string representation of behavioural performance.
#     This includes measures of choice point turns and of retrieval trial performance."""

#     string = '\n'.join([
#         'Completed all trials: {}'.format(str(behav_info['complete_session'])),
#         'Choice point - correct turns: {} / {} ({:5.2%})'.format(\
#             behav_info['turn_correct'], behav_info['turn_total'],
#             behav_info['turn_correct'] / behav_info['turn_total']),
#         '    encoding : {} / {} ({:5.2%})'.format(\
#             behav_info['cp_enc_cor'], behav_info['cp_enc_tot'], behav_info['cp_enc_perc']),
#         '    retrieval: {} / {} ({:5.2%})'.format(\
#             behav_info['cp_ret_cor'], behav_info['cp_ret_tot'], behav_info['cp_ret_perc']),
#         'Number of retrieval responses: ({} / {})'.format(\
#             behav_info['n_responses'], behav_info['n_retrieval']),
#         'Retrieval error (correct turn): {:4.2f}'.format(behav_info['err_all']),
#         '    left : {:5.2f}'.format(behav_info['err_left']),
#         '    right: {:5.2f}'.format(behav_info['err_right']),

#     ])

#     return string


## UNIT

def create_unit_info(unit):
    """Create a dictionary of unit information."""

    spikes = unit['spike_times'].values[0]

    unit_info = {}

    unit_info['n_spikes'] = len(spikes)
    unit_info['firing_rate'] = compute_firing_rate(spikes)
    unit_info['first_spike'] = spikes[0]
    unit_info['last_spike'] = spikes[-1]
    #unit_info['hemisphere'] = unit['hemisphere'].values[0]
    #unit_info['location'] = unit['location'].values[0]
    unit_info['channel'] = unit['channel'].values[0]

    return unit_info


def create_unit_str(unit_info):
    """Create a string representation of the unit info."""

    string = '\n'.join([
        '# spikes:    {:10d}'.format(unit_info['n_spikes']),
        'firing rate:  {:10.2f}'.format(unit_info['firing_rate']),
        #'location:    {}'.format(unit_info['location']),
        'channel:    {}'.format(unit_info['channel']),
        #'cluster:    {}'.format(unit_info['cluster']),

        'Recording time range: {:5.4f} - {:5.4f}'.format(\
           unit_info['first_spike'], unit_info['last_spike'])
    ])

    return string