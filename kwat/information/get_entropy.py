from numpy import log


def get_entropy(nu_):

    pr_ = nu_ / nu_.sum()

    return -(pr_ * log(pr_)).sum()
