from numpy import absolute, log2, sign


def log_with_negative(ar):

    return sign(ar) * log2(absolute(ar) + 1)
