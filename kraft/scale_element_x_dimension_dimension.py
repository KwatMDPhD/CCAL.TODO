from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

from .RANDOM_SEED import RANDOM_SEED


def scale_element_x_dimension_dimension(
    n_target_dimension,
    point_x_dimension=None,
    distance__point_x_point=None,
    distance_function="euclidean",
    metric=True,
    n_init=int(1e3),
    max_iter=int(1e3),
    verbose=0,
    eps=1e-3,
    n_job=1,
    random_seed=RANDOM_SEED,
):

    keyword_arguments = {
        "n_components": n_target_dimension,
        "metric": metric,
        "n_init": n_init,
        "max_iter": max_iter,
        "verbose": verbose,
        "eps": eps,
        "n_jobs": n_job,
        "random_state": random_seed,
    }

    if distance__point_x_point is None and not callable(distance_function):

        mds = MDS(dissimilarity=distance_function, **keyword_arguments)

        point_x_target_dimension = mds.fit_transform(point_x_dimension)

    else:

        mds = MDS(dissimilarity="precomputed", **keyword_arguments)

        if distance__point_x_point is None:

            distance__point_x_point = squareform(
                pdist(point_x_dimension, distance_function)
            )

        point_x_target_dimension = mds.fit_transform(distance__point_x_point)

    return point_x_target_dimension
