from numpy import logical_and

from ..array import check_not_nan


def apply(nu1___, nu2___, fu):
    go___ = logical_and(check_not_nan(nu1___), check_not_nan(nu2___))

    return fu(nu1___[go___], nu2___[go___])
