from ..array import apply as array_apply, normalize


def _process_target(ta, ty, st):

    if ty == "continuous":

        if 0 < ta.std():

            ta = array_apply(ta, normalize, "-0-", up=True).clip(-st, st)

        return ta, -st, st

    return ta.copy(), None, None
