from numpy import array


def make_factor_label_(
    r,
):

    return array(tuple("Factor {}_{}".format(r, index) for index in range(r)))
