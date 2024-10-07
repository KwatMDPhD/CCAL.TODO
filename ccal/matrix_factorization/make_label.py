from pandas import Index


def make_label(re):
    na = "Factor"

    return Index(data=("{} {} {}".format(na, re, ie + 1) for ie in range(re)), name=na)
