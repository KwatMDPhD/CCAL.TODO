from numpy import concatenate, where
from pandas import unique
from scipy.cluster.hierarchy import leaves_list, linkage

from .ignore_nan_and_function_2 import ignore_nan_and_function_2


def get_clustering_index(
    matrix,
    axis,
    groups=None,
    linkage_metric=None,
    linkage_method="average",
    optimal_ordering=True,
):

    if axis == 1:

        matrix = matrix.T

    if linkage_metric is None:

        def linkage_metric(vector_0, vector_1):

            return ignore_nan_and_function_2(
                vector_0,
                vector_1,
                lambda vector_0, vector_1: ((vector_0 - vector_1) ** 2).sum() ** 0.5,
            )

    def get_linkage_leaves(matrix):

        return leaves_list(
            linkage(
                matrix,
                metric=linkage_metric,
                method=linkage_method,
                optimal_ordering=optimal_ordering,
            )
        )

    if groups is None:

        return get_linkage_leaves(matrix)

    else:

        index = []

        for group in unique(groups):

            group_index = where(groups == group)[0]

            clustered_index = get_linkage_leaves(matrix[group_index, :])

            index.append(group_index[clustered_index])

        return concatenate(index)
