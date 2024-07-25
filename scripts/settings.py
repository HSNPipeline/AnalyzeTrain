"""Settings for running analysis scripts."""

from pathlib import Path

###################################################################################################
## RUN SETTINGS

# Set which task to process
TASK = 'Train'

# Set files to ignore
IGNORE = []

# Set verboseness
VERBOSE = True

RUN = {
    'TASK' : TASK,
    'IGNORE' : IGNORE,
    'VERBOSE' : VERBOSE,
}

###################################################################################################
## PATHS

# Set the data path to load from
BASE_PATH = Path('/Users/weijiazhang/Data/Train')
DATA_PATH = BASE_PATH / 'nwb'

# Set the path to save out reports & results
REPORTS_PATH = Path(BASE_PATH/'reports')
RESULTS_PATH = Path(BASE_PATH/'results')

PATHS = {
    'BASE' : BASE_PATH,
    'DATA' : DATA_PATH,
    'REPORTS' : REPORTS_PATH,
    'RESULTS' : RESULTS_PATH
}

###################################################################################################
## UNIT SETTINGS

# Set whether to skip units that have already been processed
SKIP_ALREADY_RUN = False
SKIP_FAILED = False
CONTINUE_ON_FAIL = True

UNITS = {
    'SKIP_ALREADY_RUN' : SKIP_ALREADY_RUN,
    'SKIP_FAILED' : SKIP_FAILED,
    'CONTINUE_ON_FAIL' : CONTINUE_ON_FAIL,
}

###################################################################################################
## METHOD SETTINGS
PLACE_METHODS = ['ANOVA','info']
METHODS = {
'PLACE': PLACE_METHODS

}

###################################################################################################
## ANALYSIS SETTINGS

...

# SURROGATE SETTINGS

SHUFFLE_APPROACH = 'isi'   # 'CIRCULAR', 'BINCIRC'
N_SHUFFLES = 100


SURROGATES = {
    'approach' : SHUFFLE_APPROACH,
    'n_shuffles' : N_SHUFFLES
}
