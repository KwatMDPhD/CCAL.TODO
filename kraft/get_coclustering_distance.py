from numpy import apply_along_axis, asarray, isnan, triu_indices
from scipy.spatial.distance import squareform


def get_coclustering_distance(clustering_x_point):

    index_0_index_1 = tuple(zip(*triu_indices(clustering_x_point.shape[1], k=1)))

    def is_coclustered(clusterings):

        return tuple(
            clusterings[index_0] == clusterings[index_1]
            for index_0, index_1 in index_0_index_1
        )

    n_coclustered = apply_along_axis(is_coclustered, 1, clustering_x_point).sum(axis=0)

    return squareform(
        1
        - n_coclustered
        / asarray(
            tuple(
                (~isnan(clustering_x_point[:, index_0_index_1_])).all(axis=1).sum()
                for index_0_index_1_ in index_0_index_1
            )
        )
    )
