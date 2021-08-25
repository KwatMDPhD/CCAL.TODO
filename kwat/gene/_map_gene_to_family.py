from ._read_select_hgnc import _read_select_hgnc
from ._split_get_first import _split_get_first


def _map_gene_to_family():

    da = _read_select_hgnc(None)

    return dict(
        zip(
            da.loc[:, "symbol"],
            (_split_get_first(fa) for fa in da.loc[:, "gene_family"]),
        )
    )
