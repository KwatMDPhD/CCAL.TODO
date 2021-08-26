from ..array import apply, normalize


def _process_target(ta, ty, st):

    if ty == "continuous":

        if 0 < ta.std():

            ta = apply(ta, normalize, "-0-", up=True).clip(min=-st, max=st)

        return ta, -st, st

    return ta.copy(), None, None
