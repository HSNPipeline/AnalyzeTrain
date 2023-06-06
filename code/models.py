""""Helper code & functions to define analysis models."""

from functools import partial

from spiketools.stats.anova import create_dataframe, create_dataframe_bins, fit_anova

###################################################################################################
###################################################################################################
PLACE = {
    'MODEL' : 'fr ~ C(bin)',
    'FEATURE' : 'C(bin)',
    'COLUMNS' : ['bin', 'fr']
}

create_df_place = partial(create_dataframe_bins)
fit_anova_place = partial(fit_anova, formula=PLACE['MODEL'], feature=PLACE['FEATURE'])
