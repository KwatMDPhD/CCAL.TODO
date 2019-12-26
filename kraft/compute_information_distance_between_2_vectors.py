from .compute_information_correlation_between_2_vectors import (
    compute_information_correlation_between_2_vectors,
)


def compute_information_distance_between_2_vectors(vector_0, vector_1):

    return (
        1 - compute_information_correlation_between_2_vectors(vector_0, vector_1)
    ) / 2
