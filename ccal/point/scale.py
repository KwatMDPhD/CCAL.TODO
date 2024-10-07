from numpy import apply_along_axis
from sklearn.manifold import MDS

from ..array import normalize
from ..constant import RANDOM_SEED


def scale(di_po_po, n_di, ra=RANDOM_SEED, **ke_ar):
    return apply_along_axis(
        normalize,
        0,
        MDS(
            n_components=n_di, random_state=ra, dissimilarity="precomputed", **ke_ar
        ).fit_transform(di_po_po),
        "0-1",
    )
