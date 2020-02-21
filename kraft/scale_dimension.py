from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

from .RANDOM_SEED import RANDOM_SEED


def scale_dimension(
    n_target_dimension,
    point_x_dimension=None,
    point_x_point_distance=None,
    distance_function="euclidean",
    metric=True,
    n_init=int(1e3),
    max_iter=int(1e3),
    verbose=0,
    eps=1e-3,
    n_job=1,
    random_seed=RANDOM_SEED,
):

    if point_x_point_distance is None:

        point_x_point_distance = squareform(
            pdist(point_x_dimension, metric=distance_function)
        )

    return MDS(
        n_components=n_target_dimension,
        metric=metric,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        eps=eps,
        n_jobs=n_job,
        random_state=random_seed,
        dissimilarity="precomputed",
    ).fit_transform(point_x_point_distance)
