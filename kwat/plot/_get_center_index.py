from numpy import nonzero


def _get_center_index(gr_, gr):

    ie1, ie2 = nonzero(gr_ == gr)[0][[0, -1]]

    return ie1 + (ie2 - ie1) / 2
