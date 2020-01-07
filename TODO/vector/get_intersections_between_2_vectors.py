from numpy import diff, insert, sign

from .is_array_bad import is_array_bad


def get_intersections_between_2_vectors(vector_0, vector_1, raise_if_bad=True):

    is_array_bad(vector_0, raise_if_bad=raise_if_bad)

    is_array_bad(vector_1, raise_if_bad=raise_if_bad)

    diff_sign = sign(vector_0 - vector_1)

    diff_sign_0_indices = (diff_sign == 0).nonzero()[0]

    if 0 < diff_sign_0_indices.size:

        diff_sign[diff_sign_0_indices] = diff_sign[diff_sign_0_indices + 1]

    return insert(diff(diff_sign) != 0, 0, False)
