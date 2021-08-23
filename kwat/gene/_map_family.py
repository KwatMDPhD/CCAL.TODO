from ._pr1 import _pr1
from ._read_hgnc import _read_hgnc


def _map_family():

    da = _read_hgnc(None)

    return dict(zip(da.loc[:, "symbol"], (_pr1(fa) for fa in da.loc[:, "gene_family"])))
