from numpy import diff

from .error_nan import error_nan


def is_sorted(array):

    error_nan(array)

    diff_ = diff(array)

    return (diff_ <= 0).all() or (0 <= diff_).all()
