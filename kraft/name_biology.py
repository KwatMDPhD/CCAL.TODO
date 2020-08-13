from numpy import asarray, full
from pandas import read_csv, read_excel

from .CONSTANT import DATA_DIRECTORY_PATH


def map_str_to_gene():

    str_to_gene = {}

    hgnc = read_csv(
        "{}/hgnc_complete_set.txt.gz".format(DATA_DIRECTORY_PATH), sep="\t", index_col=1
    )

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

                    str_to_gene[split] = gene

    return str_to_gene


def map_cg_to_gene():

    cg_to_gene = {}

    for cg_to_genes in (
        read_excel(
            "{}/illumina_humanmethylation27_content.xlsx".format(DATA_DIRECTORY_PATH),
            usecols=(0, 10),
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            "{}/HumanMethylation450_15017482_v1-2.csv.gz".format(DATA_DIRECTORY_PATH),
            skiprows=7,
            usecols=(0, 21),
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            "{}/infinium-methylationepic-v-1-0-b5-manifest-file-csv.zip".format(
                DATA_DIRECTORY_PATH
            ),
            skiprows=7,
            usecols=(0, 15),
            index_col=0,
            squeeze=True,
        ),
    ):

        for cg, genes in cg_to_genes.dropna().items():

            cg_to_gene[cg] = genes.split(sep=";", maxsplit=1)[0]

    return cg_to_gene


def name_genes(ids):

    _to_gene = {**map_str_to_gene(), **map_cg_to_gene()}

    genes = asarray(tuple(_to_gene.get(id_) for id_ in ids))

    is_named = asarray(tuple(gene is not None for gene in genes))

    n = is_named.size

    n_named = is_named.sum()

    print("Named {}/{} ({:.2%})".format(n_named, n, n_named / n))

    if n_named == 0:

        return ids

    else:

        return genes


def map_cell_line_name_to_rename():

    return read_csv(
        "{}/cell_line_name_rename.tsv.gz".format(DATA_DIRECTORY_PATH),
        sep="\t",
        index_col=0,
        squeeze=True,
    ).to_dict()


def name_cell_lines(names):

    name_to_rename = map_cell_line_name_to_rename()

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

        print("Failed to name: {}.".format(sorted(set(fails))))

    return asarray(renames)


def select_genes(selection=None):

    if selection is None:

        selection = {"locus_group": ("protein-coding gene",)}

    hgnc = read_csv(
        "{}/hgnc_complete_set.txt.gz".format(DATA_DIRECTORY_PATH), sep="\t", index_col=1
    )

    genes = hgnc.index.to_numpy()

    is_selected = full(genes.size, True)

    for label, selection in selection.items():

        print("Selecting by {}...".format(label))

        is_selected &= asarray(
            tuple(
                isinstance(value, str) and value in selection
                for value in hgnc.loc[:, label].to_numpy()
            )
        )

        print("{}/{}".format(is_selected.sum(), is_selected.size))

    return genes[is_selected]
