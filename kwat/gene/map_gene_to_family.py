from ..string import split_and_get
from ._read import _read


def map_gene_to_family():

    da = _read({})

    return dict(
        zip(
            da.loc[:, "symbol"].values,
            (split_and_get(fa, "|", 0) for fa in da.loc[:, "gene_family"].values),
        )
    )
