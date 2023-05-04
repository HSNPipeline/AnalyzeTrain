"""Local utility functions."""

###################################################################################################
###################################################################################################
def group_array_by_key(keys, values):
    result = {}
    for key, value in zip(keys, values):
        if key not in result:
            result[key] = []
        result[key].append(value)
    return result