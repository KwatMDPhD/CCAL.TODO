from .apply_function_on_2_vectors import apply_function_on_2_vectors


def ignore_bad_and_compute_euclidean_distance(vector_0, vector_1):

    return apply_function_on_2_vectors(
        vector_0,
        vector_1,
        lambda vector_0, vector_1: ((vector_0 - vector_1) ** 2).sum() ** 0.5,
        raise_for_bad=False,
    )
