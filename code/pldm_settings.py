
### Feature Names
PEAK = 'Peak Rate'
AVERAGE = 'Average Rate'
PEAK_OVER_AVERAGE = 'Peak Over Average'
PLACE_FIELD_WIDTH = 'Place Field Width'
N_PLACE_FIELD = 'Num Place Fields'
PLACE_FIELD_CONSISTENCY = 'Place Field Consistency'
PRESENCE_RATIO = 'Presence Ratio'
EVEN_ODD_CORRELATION = 'Even Odd Correlation'

FEATURES = [PEAK, AVERAGE, PEAK_OVER_AVERAGE, PLACE_FIELD_WIDTH, N_PLACE_FIELD, PLACE_FIELD_CONSISTENCY, PRESENCE_RATIO, EVEN_ODD_CORRELATION]

## METHODS
SI_THRESHOLD = 0.25
ALPHA = 0.05

##
PLACE_FIELD_THRESH = .2
PLACE_FIELD_NOISE_THRESH = .2
TOLERANCE = 3 ## number of spatial bins away from the peak location to be considered a place field

## COLORS
ANOVA_COLOR = '#002FA7'
INFO_COLOR = '#D92911'
SI_PERM_COLOR = '#610200'
SI_THRESH_COLOR = INFO_COLOR
THRESHOLD_COLOR = '#000000'

COLORS_METHOD = {
    'ANOVA': ANOVA_COLOR,
    'INFO':INFO_COLOR,
    'SI_PERM':SI_PERM_COLOR,
    'SI_THRESH':SI_THRESH_COLOR,
    'THRESHOLD':THRESHOLD_COLOR
}



PLOT_PARAMS = {
    'font.family': 'Avenir',
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,  # Smaller tick label size
    'ytick.labelsize': 10,  # Smaller tick label size
    'axes.linewidth': 0.5,  # Reduce the thickness of axis lines
    'xtick.major.width': 0.8,  # Thinner x ticks
    'ytick.major.width': 0.8,  # Thinner y ticks
    'xtick.major.size': 2.0,   # Shorter x ticks
    'ytick.major.size': 2.0    # Shorter y ticks
}


FEATURE_COLORS = {
    PEAK_OVER_AVERAGE: '#ff3b2d',
    PLACE_FIELD_CONSISTENCY: '#ff9500', 
    EVEN_ODD_CORRELATION: '#ffdd00',
    PEAK: '#00c78c',
    PRESENCE_RATIO: '#00d6ff',
    AVERAGE: '#007aff',
    PLACE_FIELD_WIDTH: '#bf5af2',
    N_PLACE_FIELD: 'brown'
}