from numpy import median


def get_median_ratio(ve0, ve1):

    return median(ve1) / median(ve0)
