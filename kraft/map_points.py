from sklearn.manifold import MDS

from .normalize import normalize
from .RANDOM_SEED import RANDOM_SEED


def map_points(
    point_x_point_distance,
    n_dimension,
    metric=True,
    n_init=int(1e3),
    max_iter=int(1e3),
    verbose=0,
    eps=1e-3,
    n_job=1,
    random_seed=RANDOM_SEED,
):

    point_x_dimension = MDS(
        n_components=n_dimension,
        metric=metric,
        n_init=n_init,
        max_iter=max_iter,
        verbose=verbose,
        eps=eps,
        n_jobs=n_job,
        random_state=random_seed,
        dissimilarity="precomputed",
    ).fit_transform(point_x_point_distance)

    return normalize(point_x_dimension, 0, "0-1")
