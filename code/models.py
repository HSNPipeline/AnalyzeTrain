import numpy as np
from functools import partial
from spiketools.stats.anova import create_dataframe, create_dataframe_bins, fit_anova
from spiketools.spatial.information import compute_spatial_information
###################################################################################################
###################################################################################################
PLACE_MODELS = {
    'MODEL' : 'fr ~ C(bin)',
    'FEATURE' : 'C(bin)',
    'COLUMNS' : ['bin', 'fr']
}

create_df_place = partial(create_dataframe_bins,bin_columns=PLACE_MODELS['COLUMNS'])
fit_anova_place = partial(fit_anova, formula=PLACE_MODELS['MODEL'], feature=PLACE_MODELS['FEATURE'])



OBJECT_MODELS = {
    #'MODEL' : 'fr ~ C(tbin) + C(side) + C(tbin):C(side)',
    #'FEATURE' : 'C(tbin):C(side)',
    'MODEL' : 'fr ~ C(tbin) + C(object)',
    'FEATURE' : 'C(tbin)',
    'COLUMNS' : ['tbin', 'fr']
}

create_df_object = partial(create_dataframe_bins, bin_columns=OBJECT_MODELS['COLUMNS'])
fit_anova_object = partial(fit_anova, formula=OBJECT_MODELS['MODEL'],
                          feature=OBJECT_MODELS['FEATURE'])


PLACE_COND_MODELS = {
    'MODEL' : 'fr ~ C(bin) + C(condition) + C(bin):C(condition)',
    'FEATURE' : 'C(bin)',
    'COLUMNS' : ['bin', 'fr', 'condition']
}

create_df_place_cond = partial(create_dataframe_bins, bin_columns=PLACE_COND_MODELS['COLUMNS'])
fit_anova_place_cond = partial(fit_anova, formula=PLACE_COND_MODELS['MODEL'],
                          feature=PLACE_COND_MODELS['FEATURE'])