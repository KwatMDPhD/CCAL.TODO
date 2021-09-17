from ..string import split_and_get
from ._read import _read


def map_gene_to_family():

    hg = _read({})

    return dict(
        zip(
            hg.loc[:, "symbol"].values,
            (split_and_get(fa, "|", 0) for fa in hg.loc[:, "gene_family"].values),
        )
    )
