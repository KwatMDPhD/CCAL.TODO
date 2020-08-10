from numpy import apply_along_axis, arange, asarray, full, isnan, nan, triu_indices
from numpy.random import choice, seed
from pandas import Series
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

from . import RANDOM_SEED
from .plot import DATA_TYPE_TO_COLORSCALE, plot_heat_map


def cluster(
    point_x_dimension,
    distance_function="euclidean",
    linkage_method="ward",
    optimal_ordering=False,
    n_cluster=None,
    criterion="maxclust",
):

    link = linkage(
        point_x_dimension,
        metric=distance_function,
        method=linkage_method,
        optimal_ordering=optimal_ordering,
    )

    leaf_is = leaves_list(link)

    if n_cluster is None:

        groups = None

    else:

        groups = fcluster(link, n_cluster, criterion=criterion) - 1

    return leaf_is, groups


def get_coclustering_distance(point_x_clustering, min_n_clustered):

    point_i_pairs = tuple(zip(*triu_indices(point_x_clustering.shape[0], k=1)))

    n_clustered = asarray(
        tuple(
            (~isnan(point_x_clustering[is_, :])).all(axis=0).sum()
            for is_ in point_i_pairs
        )
    )

    assert (min_n_clustered < n_clustered).all()

    def check_is_coclustered(clusters, point_i_pairs):

        return tuple(clusters[i_0] == clusters[i_1] for i_0, i_1 in point_i_pairs)

    n_coclustered = apply_along_axis(
        check_is_coclustered, 0, point_x_clustering, point_i_pairs
    ).sum(axis=1)

    return squareform(1 - n_coclustered / n_clustered)


def cluster_hierarchical_clusterings(
    point_x_dimension,
    n_cluster,
    n_clustering=100,
    random_seed=RANDOM_SEED,
    min_n_clustered=0,
    plot=True,
    **cluster_keyword_arguments,
):

    matrix = point_x_dimension.to_numpy()

    n_point = matrix.shape[0]

    point_x_clustering = full((n_point, n_clustering), nan)

    point_is_ = arange(n_point)

    n_choice = int(n_point * 0.632)

    seed(seed=random_seed)

    n_per_print = max(1, n_clustering // 10)

    for i in range(n_clustering):

        if i % n_per_print == 0:

            print("{}/{}...".format(i + 1, n_clustering))

        is_ = choice(point_is_, size=n_choice, replace=False)

        point_x_clustering[is_, i] = cluster(
            matrix[is_], n_cluster=n_cluster, **cluster_keyword_arguments
        )[1]

    leaf_is, clusters = cluster(
        get_coclustering_distance(point_x_clustering, min_n_clustered),
        n_cluster=n_cluster,
        **cluster_keyword_arguments,
    )

    if plot:

        plot_heat_map(
            point_x_dimension.iloc[leaf_is, :],
            sort_groups=False,
            layout={"title": {"text": "Clustering {}".format(n_cluster)}},
            **{
                "axis_0_groups": clusters[leaf_is],
                "axis_0_group_colorscale": DATA_TYPE_TO_COLORSCALE["categorical"],
            },
        )

    return Series(clusters, index=point_x_dimension.index, name="Group")
