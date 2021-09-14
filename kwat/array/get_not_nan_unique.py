from numpy import unique

from .check_not_nan import check_not_nan


def get_not_nan_unique(ar):

    return unique(ar[check_not_nan(ar)])
