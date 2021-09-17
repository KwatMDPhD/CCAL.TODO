from numpy import apply_along_axis

from ._process_target import _process_target


def _process_target_and_get_1(tav, ty, st):

    return _process_target(tav, ty, st)[0]


def _process_data(dav, ty, st):

    dav = apply_along_axis(_process_target_and_get_1, 1, dav, ty, st)

    if ty == "continuous":

        return dav, -st, st

    else:

        return dav, None, None
