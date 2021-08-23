from numpy import log


def get_entropy(ve):

    pr_ = ve / ve.sum()

    return -(pr_ * log(pr_)).sum()
