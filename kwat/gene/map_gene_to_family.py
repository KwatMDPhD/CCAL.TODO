from ._read import _read
from ._split import _split


def _split_get_first(an):

    sp_ = _split(an)

    if 0 < len(sp_):

        return sp_[0]

    else:

        return None


def map_gene_to_family():

    da = _read(None)

    return dict(
        zip(
            da.loc[:, "symbol"].values,
            (_split_get_first(fa) for fa in da.loc[:, "gene_family"].values),
        )
    )
