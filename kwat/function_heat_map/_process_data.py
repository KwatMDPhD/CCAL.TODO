from ._process_target import _process_target


def _process_data(da, ty, st):

    da = da.copy()

    if ty == "continuous":

        for ie in range(da.shape[0]):

            da[ie] = _process_target(da[ie], ty, st)[0]

        return da, -st, st

    return da, None, None
