from numpy import full, nan

from .check_not_nan import check_not_nan


def apply(nu___, fu, *ar_, up=False, **ke_va):

    go___ = check_not_nan(nu___)

    nu = fu(nu___[go___], *ar_, **ke_va)

    if up:

        nu2___ = full(nu___.shape, nan)

        nu2___[go___] = nu

        return nu2___

    else:

        return nu
