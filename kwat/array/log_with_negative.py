from numpy import absolute, sign

from .log import log


def log_with_negative(ar, *arg, **ke):

    return sign(ar) * log(absolute(ar) + 1, *arg, **ke)
