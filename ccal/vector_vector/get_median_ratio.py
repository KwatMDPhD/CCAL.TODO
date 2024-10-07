from numpy import median


def get_median_ratio(ve1, ve2):
    return median(ve2) / median(ve1)
