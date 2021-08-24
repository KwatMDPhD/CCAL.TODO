from pandas import notna


def _check_has_enough_not_na(ve, n_no):

    return n_no <= notna(ve).sum()
