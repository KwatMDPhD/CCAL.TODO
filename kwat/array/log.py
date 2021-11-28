from numpy import absolute
from numpy import log as loge
from numpy import log2, log10, sign

from .shift import shift


def log(nu___, ba=2, ab=False, sh=None):

    lo = {2: log2, "e": loge, 10: log10}[ba]

    if ab:

        return sign(nu___) * lo(shift(absolute(nu___), "1<"))

    elif sh is not None:

        return lo(shift(nu___, sh))

    else:

        return lo(nu___)
