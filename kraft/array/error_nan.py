from numpy import isnan


def error_nan(array):

    assert not isnan(array).any()
