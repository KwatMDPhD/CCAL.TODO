from numpy import concatenate, where
from pandas import unique
from scipy.cluster.hierarchy import dendrogram, linkage

from .check_array_for_bad import check_array_for_bad
from .ignore_bad_and_compute_euclidean_distance import (
    ignore_bad_and_compute_euclidean_distance,
)


def cluster_matrix(
    matrix,
    axis,
    groups=None,
    distance_function=ignore_bad_and_compute_euclidean_distance,
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
