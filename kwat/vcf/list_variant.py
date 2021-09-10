from ._get_info_ann import _get_info_ann
from .columns import columns


def list_variant(se):

    io = se[columns.index("INFO")]

    return set(
        "{} ({})".format(ge, ef)
        for ge, ef in zip(_get_info_ann(io, "gene_name"), _get_info_ann(io, "effect"))
    )
