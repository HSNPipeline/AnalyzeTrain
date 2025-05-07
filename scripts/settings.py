"""Settings for running analysis scripts."""

from pathlib import Path

###################################################################################################
## RUN SETTINGS

# Set which task to process
TASK = 'Train'

# Set files to ignore
IGNORE = ['TRAIN_R1027J_session_0.nwb',
          'TRAIN_R1027J_session_1.nwb',
          'TRAIN_R1030J_session_0.nwb',
          'TRAIN_R1030J_session_2.nwb' ]

# Run specific 
RUN_SPECIFIC = [
]

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
DATA_PATH = BASE_PATH / 'data_matfile'

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
PLACE_METHODS = ['ANOVA','INFO']
METHODS = {
'PLACE': PLACE_METHODS,
'PLACE_FR': PLACE_METHODS


}

###################################################################################################
## ANALYSIS SETTINGS
## SPATIAL BIN SETTINGS

PLACE_BINS = 40
# DISTANCE_BINS = [47]
# NBINS_STEM = 4
# STEM_BIN_RANGE = [-131, -35]

BINS = {
    'place' : PLACE_BINS,
#     'distance' : DISTANCE_BINS,
#     'nbins_stem' : NBINS_STEM,
#     'stem_bin_range' : STEM_BIN_RANGE,
}

EXCLUSION = {
    'zscore': 3.29

}
## OCCUPANCY & PLACE BIN SETTINGS

OCC_MINIMUM = .1
OCC_SETNAN = True

MIN_SPEED = 2
MAX_TIME = 0.25
TIME_THRESHOLD = .1

## Note: min_time and max_time is different from minimim 
OCCUPANCY = {
    'minimum' : OCC_MINIMUM,
    'set_nan' : OCC_SETNAN,
    'min_speed' : MIN_SPEED,
    'max_time' : MAX_TIME,
}

OCCUPANCY_TRIAL = {
    'set_nan' : OCC_SETNAN,
    'min_speed' : MIN_SPEED,
    'max_time' : MAX_TIME,
}


PLACE = {
    'min_speed' : MIN_SPEED,
    'time_threshold' :TIME_THRESHOLD,
}


# SURROGATE SETTINGS

SHUFFLE_APPROACH = 'ISI'   # 'CIRCULAR', 'BINCIRC'
N_SHUFFLES = 1000

 
SURROGATES = {
    'approach' : SHUFFLE_APPROACH,
    'n_shuffles' : N_SHUFFLES
}
