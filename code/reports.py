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