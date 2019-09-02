from numpy import diff

from .check_array_for_bad import check_array_for_bad


def is_sorted_array(array, raise_for_bad=True):

    check_array_for_bad(array, raise_for_bad=raise_for_bad)

    diff_ = diff(array)

    return (diff_ <= 0).all() or (0 <= diff_).all()
