from pandas import Index


def label_factor(re):

    na = "Factor"

    return Index(data=("{} {}_{}".format(na, re, ie) for ie in range(re)), name=na)
