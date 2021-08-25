from ..dataframe import map_to
from ._read_select_hgnc import _read_select_hgnc
from ._split import _split


def _map_hgnc_to_gene():
    return map_to(
        _read_select_hgnc(None).drop(
            labels=[
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
        fu=_split,
    )
