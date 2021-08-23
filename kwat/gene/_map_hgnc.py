from ..dataframe import map_to
from ._pr import _pr
from ._read_hgnc import _read_hgnc


def _map_hgnc():
    return map_to(
        _read_hgnc(None).drop(
            [
                "locus_group",
                "locus_type",
                "status",
                "location",
                "location_sortable",
                "gene_family",
                "gene_family_id",
                "date_approved_reserved",
                "date_symbol_changed",
                "date_name_changed",
                "date_modified",
                "pubmed_id",
                "lsdb",
            ],
            axis=1,
        ),
        "symbol",
        fu=_pr,
    )
