from numpy import absolute, log as loge, log2, log10, sign

from .shift import shift


def log(ar, mi=None, ba=2):

    lo = {
        2: log2,
        "e": loge,
        10: log10,
    }[ba]

    if mi is not None:

        if mi < 0:

            return sign(ar) * lo(shift(absolute(ar), "0<"))

        else:

            ar = shift(ar, mi)

    return lo(ar)
