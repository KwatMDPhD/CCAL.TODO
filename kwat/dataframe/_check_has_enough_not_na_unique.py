from numpy import unique
from pandas import notna


def _check_has_enough_not_na_unique(ve, n_un):

    return n_un <= unique(ve[notna(ve)]).size
