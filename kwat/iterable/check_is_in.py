from numpy import array


def check_is_in(an1_, an2_):

    an2_no = {an2: None for an2 in an2_}

    return array([an1 in an2_no for an1 in an1_], bool)
