from ..array import apply, normalize


def _process_target(tav, ty, st):

    if ty == "continuous":

        if 0 < tav.std():

            tav = apply(tav, normalize, "-0-", up=True).clip(min=-st, max=st)

        return tav, -st, st

    else:

        return tav.copy(), None, None
