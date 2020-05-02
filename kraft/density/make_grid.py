from numpy import linspace


def make_grid(min_, max_, fraction_extension, n):

    if 0 < fraction_extension:

        extension = (max_ - min_) * fraction_extension

        min_ -= extension

        max_ += extension

    return linspace(min_, max_, num=n)
