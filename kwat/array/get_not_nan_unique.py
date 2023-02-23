from numpy import unique

from .check_not_nan import check_not_nan


def get_not_nan_unique(nu___):
    return unique(nu___[check_not_nan(nu___)])
