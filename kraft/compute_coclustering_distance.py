from numpy import isnan, triu_indices, zeros
from scipy.spatial.distance import squareform


def compute_coclustering_distance(clustering_x_point):

    index_0_index_1 = tuple(zip(*triu_indices(clustering_x_point.shape[1], k=1)))

    point_x_point = zeros(len(index_0_index_1))

    for clusterings in clustering_x_point:

        for i, (index_0, index_1) in enumerate(index_0_index_1):

            point_x_point[i] += int(clusterings[index_0] == clusterings[index_1])

    n_clustering = clustering_x_point.shape[0]

    for i, index_0_index_1_ in enumerate(index_0_index_1):

        n_for_clustering = (
            n_clustering
            - isnan(clustering_x_point[:, index_0_index_1_]).any(axis=1).sum()
        )

        point_x_point[i] /= n_for_clustering

    point_x_point = squareform(1 - point_x_point)

    point_x_point[isnan(point_x_point)] = 1

    return point_x_point
