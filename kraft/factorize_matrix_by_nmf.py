from sklearn.decomposition import NMF

from .RANDOM_SEED import RANDOM_SEED


def factorize_matrix_by_nmf(
    v, r, solver="cd", tolerance=1e-6, n_iteration=int(1e3), random_seed=RANDOM_SEED
):

    model = NMF(
        n_components=r,
        solver=solver,
        tol=tolerance,
        max_iter=n_iteration,
        random_state=random_seed,
    )

    w = model.fit_transform(v)

    h = model.components_

    error = model.reconstruction_err_

    return w, h, error
