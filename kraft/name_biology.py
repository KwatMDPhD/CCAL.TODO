from numpy import asarray, full
from pandas import read_csv

from .CONSTANT import DATA_DIRECTORY_PATH


def select_genes(selection=None):

    if selection is None:

        selection = {"locus_group": ("protein-coding gene",)}

    table = read_csv(
        "{}/hgnc_complete_set.txt.gz".format(DATA_DIRECTORY_PATH), sep="\t", index_col=1
    )

    genes = table.index.to_numpy()

    is_selected = full(genes.size, True)

    for label, selection in selection.items():

        print("Selecting by {}...".format(label))

        is_selected &= asarray(
            tuple(
                isinstance(value, str) and value in selection
                for value in table.loc[:, label].to_numpy()
            )
        )

        print("{}/{}".format(is_selected.sum(), is_selected.size))

    return genes[is_selected]


def name_genes(ids):

    hgnc = read_csv(
        "{}/hgnc_complete_set.txt.gz".format(DATA_DIRECTORY_PATH), sep="\t", index_col=1
    )

    _to_gene = {}

    for gene, row in zip(
        hgnc.index.to_numpy(),
        hgnc.drop(
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
        ).to_numpy(),
    ):

        for str_ in row:

            if isinstance(str_, str):

                for split in str_.split(sep="|"):

                    _to_gene[split] = gene

    genes = tuple(_to_gene.get(id_) for id_ in ids)

    is_none = tuple(gene is None for gene in genes)

    if all(is_none):

        print("Failed to name genes; returning IDs...")

        return ids

    else:

        n_gene = len(is_none) - sum(is_none)

        print("Named {} genes.".format(n_gene))

        return genes


def name_cell_lines(names):

    name_to_rename = read_csv(
        "{}/cell_line_name_rename.tsv.gz".format(DATA_DIRECTORY_PATH),
        sep="\t",
        index_col=0,
        squeeze=True,
    ).to_dict()

    renames = []

    fails = []

    for name in names:

        if isinstance(name, str):

            name_lower = name.lower()

            if name_lower in name_to_rename:

                renames.append(name_to_rename[name_lower])

            else:

                renames.append(name)

                fails.append(name)

        else:

            renames.append(None)

    if 0 < len(fails):

        print("Failed to name cell lines: {}.".format(sorted(set(fails))))

    return tuple(renames)
