from numpy import concatenate, where
from pandas import unique
from scipy.cluster.hierarchy import dendrogram, linkage

from .apply_function_on_2_vectors import apply_function_on_2_vectors
from .check_array_for_bad import check_array_for_bad


def _ignore_bad_and_compute_euclidean_distance_between_2_vectors(vector_0, vector_1):

    return apply_function_on_2_vectors(
        vector_0,
        vector_1,
        lambda vector_0, vector_1: ((vector_0 - vector_1) ** 2).sum() ** 0.5,
        raise_for_bad=False,
    )


def cluster_matrix(
    matrix,
    axis,
    groups=None,
    distance_function=_ignore_bad_and_compute_euclidean_distance_between_2_vectors,
    linkage_method="average",
    optimal_ordering=True,
    raise_for_bad=True,
):

    check_array_for_bad(matrix, raise_for_bad=raise_for_bad)

    if axis == 1:

        matrix = matrix.T

    def _linkage_dendrogram_leaves(matrix):

        return dendrogram(
            linkage(
                matrix,
                metric=distance_function,
                method=linkage_method,
                optimal_ordering=optimal_ordering,
            ),
            no_plot=True,
        )["leaves"]

    if groups is None:

        return _linkage_dendrogram_leaves(matrix)

    else:

        indices = []

        for group in unique(groups):

            group_indices = where(groups == group)[0]

            clustered_indices = _linkage_dendrogram_leaves(matrix[group_indices, :])

            indices.append(group_indices[clustered_indices])

        return concatenate(indices)
