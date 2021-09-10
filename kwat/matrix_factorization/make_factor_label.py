from numpy import array


def make_factor_label(re):

    return array(["Factor {}_{}".format(re, ie) for ie in range(re)])
