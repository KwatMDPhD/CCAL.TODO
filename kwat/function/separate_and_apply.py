from numpy import where


def separate_and_apply(ar, arb, fu):

    return fu(ar[where(arb == 0)[0]], ar[where(arb == 1)[0]])
