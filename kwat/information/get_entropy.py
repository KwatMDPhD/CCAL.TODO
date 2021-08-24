from numpy import log


def get_entropy(ve):

    pr = ve / ve.sum()

    return -(pr * log(pr)).sum()
