from numpy import where


def separate_and_apply(bo_, ar, fu):

    return fu(ar[where(bo_ == 0)[0]], ar[where(bo_ == 1)[0]])
