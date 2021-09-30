from sklearn.decomposition import NMF

from ..constant import RANDOM_SEED


def factorize_with_nmf(ma, re, so="cd", to=1e-6, n_it=int(1e3), ra=RANDOM_SEED):

    nm = NMF(
        n_components=re,
        init="random",
        solver=so,
        tol=to,
        max_iter=n_it,
        random_state=ra,
    )

    return nm.fit_transform(ma), nm.components_, nm.reconstruction_err_
