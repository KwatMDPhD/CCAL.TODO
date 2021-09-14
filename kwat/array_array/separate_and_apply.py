from numpy import where

from .apply import apply


def separate_and_apply(bo_, ar, fu):

    return apply(ar[where(bo_ == 0)[0]], ar[where(bo_ == 1)[0]], fu)
