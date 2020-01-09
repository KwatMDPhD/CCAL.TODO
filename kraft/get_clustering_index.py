from numpy import concatenate, where
from pandas import unique
from scipy.cluster.hierarchy import dendrogram, linkage

from .function_ignoring_nan import function_ignoring_nan


def get_clustering_index(
    matrix,
    axis,
    groups=None,
    distance_function=None,
    linkage_method="average",
    optimal_ordering=True,
):

    if axis == 1:

        matrix = matrix.T

    if distance_function is None:

        def distance_function(vector_0, vector_1):

            return function_ignoring_nan(
                vector_0,
                vector_1,
                lambda vector_0, vector_1: ((vector_0 - vector_1) ** 2).sum() ** 0.5,
            )

    def get_linkage_dendrogram_leaves(matrix):

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

        return get_linkage_dendrogram_leaves(matrix)

    else:

        index = []

        for group in unique(groups):

            group_index = where(groups == group)[0]

            clustered_index = get_linkage_dendrogram_leaves(matrix[group_index, :])

            index.append(group_index[clustered_index])

        return concatenate(index)
