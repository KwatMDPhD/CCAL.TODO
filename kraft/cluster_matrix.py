from numpy import concatenate, where
from pandas import unique
from scipy.cluster.hierarchy import dendrogram, linkage

from .apply_function_on_2_vectors import apply_function_on_2_vectors
from .check_array_for_bad import check_array_for_bad


def _compute_euclidean_distance_between_2_vectors(vector_0, vector_1):

    return apply_function_on_2_vectors(
        vector_0,
        vector_1,
        lambda vector_0, vector_1: ((vector_0 - vector_1) ** 2).sum()
        ** 0.5,
        raise_for_bad=False,
    )


def cluster_matrix(
    matrix,
    axis,
    groups=None,
    distance_function=None,
    linkage_method="average",
    optimal_ordering=True,
    raise_for_bad=True,
):

    check_array_for_bad(matrix, raise_for_bad=raise_for_bad)

    if axis == 1:

        matrix = matrix.T

    if distance_function is None:

        distance_function = _compute_euclidean_distance_between_2_vectors

    if groups is None:

        return dendrogram(
            linkage(
                matrix,
                method=linkage_method,
                metric=distance_function,
                optimal_ordering=optimal_ordering,
            ),
            no_plot=True,
        )["leaves"]

    else:

        if len(groups) != matrix.shape[0]:

            raise ValueError(
                f"len(groups) ({len(groups)}) != len(axis-{axis} slices) ({matrix.shape[0]})"
            )

        indices = []

        for group in unique(groups):

            group_indices = where(groups == group)[0]

            clustered_indices = dendrogram(
                linkage(
                    matrix[group_indices, :],
                    method=linkage_method,
                    metric=distance_function,
                    optimal_ordering=optimal_ordering,
                ),
                no_plot=True,
            )["leaves"]

            indices.append(group_indices[clustered_indices])

        return concatenate(indices)
