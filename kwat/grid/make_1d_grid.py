from numpy import linspace


def make_1d_grid(lo, hi, fr, n_co):
    ex = (hi - lo) * fr

    lo -= fr

    hi += fr

    return linspace(lo, hi, num=n_co)
