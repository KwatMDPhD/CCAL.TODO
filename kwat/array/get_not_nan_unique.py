from numpy import unique
from .check_is_not_nan import check_is_not_nan


def get_not_nan_unique(ar):

    return unique(ar[check_is_not_nan(ar)])
