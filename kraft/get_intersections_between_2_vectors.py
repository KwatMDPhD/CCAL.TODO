from numpy import diff, insert, sign

from .check_array_for_bad import check_array_for_bad


def get_intersections_between_2_vectors(vector_0, vector_1, raise_for_bad=True):

    check_array_for_bad(vector_0, raise_for_bad=raise_for_bad)

    check_array_for_bad(vector_1, raise_for_bad=raise_for_bad)

    diff_sign = sign(vector_0 - vector_1)

    diff_sign_0_indices = (diff_sign == 0).nonzero()[0]

    if diff_sign_0_indices.size:

        diff_sign[diff_sign_0_indices] = diff_sign[diff_sign_0_indices + 1]

    return insert(diff(diff_sign) != 0, 0, False)
