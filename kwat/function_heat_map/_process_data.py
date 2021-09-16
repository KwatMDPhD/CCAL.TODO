from numpy import apply_along_axis

from ._process_target import _process_target


def _process_target_and_get_1(nu_, ty, st):

    return _process_target(nu_, ty, st)[0]


def _process_data(nu_fe_sa, ty, st):

    nu_fe_sa = apply_along_axis(_process_target_and_get_1, 1, nu_fe_sa, ty, st)

    if ty == "continuous":

        return nu_fe_sa, -st, st

    else:

        return nu_fe_sa, None, None
