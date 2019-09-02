from .compute_information_coefficient_between_2_1d_arrays import (
    compute_information_coefficient_between_2_1d_arrays,
)


def compute_information_distance_between_2_1d_arrays(vector_0, vector_1):

    return (
        1 - compute_information_coefficient_between_2_1d_arrays(vector_0, vector_1)
    ) / 2
