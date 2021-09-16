from ..array import apply, normalize


def _process_target(nu_, ty, st):

    if ty == "continuous":

        if 0 < nu_.std():

            nu_ = apply(nu_, normalize, "-0-", up=True).clip(min=-st, max=st)

        return nu_, -st, st

    else:

        return nu_.copy(), None, None
