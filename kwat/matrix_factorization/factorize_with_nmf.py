from sklearn.decomposition import NMF

from ..constant import random_seed


def factorize_with_nmf(ma, re, so="cd", to=1e-6, n_it=int(1e3), ra=random_seed):

    model = NMF(
        n_components=re,
        solver=so,
        tol=to,
        max_iter=n_it,
        random_state=ra,
    )

    return model.fit_transform(ma), model.components_, model.reconstruction_err_
