from numpy import isnan, logical_not


def check_not_nan(ar):

    return logical_not(isnan(ar))
