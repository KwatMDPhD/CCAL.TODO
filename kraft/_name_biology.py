from numpy import (
    asarray,
    full,
)
from pandas import (
    read_csv,
    read_excel,
)

from .CONSTANT import (
    DATA_DIRECTORY_PATH,
)


def map_to_gene():

    _to_gene = {}

    hgnc = read_csv(
        "{}hgnc_complete_set.txt.gz".format(DATA_DIRECTORY_PATH),
        sep="\t",
        index_col=1,
        low_memory=False,
    )

    for (gene, row,) in zip(
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

        for value in row:

            if isinstance(
                value,
                str,
            ):

                for split in value.split(sep="|"):

                    _to_gene[split] = gene

    return _to_gene


def map_enst_to_gene():

    df = read_csv(
        "{}enst_to_gene.tsv".format(DATA_DIRECTORY_PATH),
        sep="\t",
    )

    gene_name_ = df["Gene name"]

    enst_to_gene = {
        **dict(
            zip(
                df["Transcript stable ID version"],
                gene_name_,
            )
        ),
        **dict(
            zip(
                df["Transcript stable ID"],
                gene_name_,
            )
        ),
    }

    return enst_to_gene


def map_cg_to_gene():

    cg_to_gene = {}

    for cg_to_gene_ in (
        read_excel(
            "{}illumina_humanmethylation27_content.xlsx".format(DATA_DIRECTORY_PATH),
            usecols=(
                0,
                10,
            ),
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            "{}HumanMethylation450_15017482_v1-2.csv.gz".format(DATA_DIRECTORY_PATH),
            skiprows=7,
            usecols=(
                0,
                21,
            ),
            index_col=0,
            squeeze=True,
        ),
        read_csv(
            "{}infinium-methylationepic-v-1-0-b5-manifest-file.csv.gz".format(
                DATA_DIRECTORY_PATH,
            ),
            skiprows=7,
            usecols=(
                0,
                15,
            ),
            index_col=0,
            squeeze=True,
        ),
    ):

        for (
            cg,
            gene_,
        ) in cg_to_gene_.dropna().items():

            cg_to_gene[cg] = gene_.split(
                sep=";",
                maxsplit=1,
            )[0]

    return cg_to_gene


def name_gene(
    id_,
):

    _to_gene = {
        **map_to_gene(),
        **map_enst_to_gene(),
        **map_cg_to_gene(),
    }

    gene_ = asarray(tuple(_to_gene.get(id_) for id_ in id_))

    is_ = asarray(tuple(gene is not None for gene in gene_))

    n = is_.size

    name_n = is_.sum()

    print(
        "Named {}/{} ({:.2%})".format(
            name_n,
            n,
            name_n / n,
        )
    )

    if name_n == 0:

        return id_

    else:

        return gene_


def map_cell_line_name_to_rename():

    return read_csv(
        "{}cell_line_name_rename.tsv.gz".format(DATA_DIRECTORY_PATH),
        sep="\t",
        index_col=0,
        squeeze=True,
    ).to_dict()


def name_cell_line(
    name_,
):

    name_to_rename = map_cell_line_name_to_rename()

    rename_ = []

    fail_ = []

    for name in name_:

        if isinstance(
            name,
            str,
        ):

            name_lower = name.lower()

            if name_lower in name_to_rename:

                rename_.append(name_to_rename[name_lower])

            else:

                rename_.append(name)

                fail_.append(name)

        else:

            rename_.append(None)

    if 0 < len(fail_):

        print("Failed to name: {}.".format(sorted(set(fail_))))

    return asarray(rename_)


def select_genes(
    selection=None,
):

    if selection is None:

        selection = {"locus_group": ("protein-coding gene",)}

    hgnc = read_csv(
        "{}hgnc_complete_set.txt.gz".format(DATA_DIRECTORY_PATH),
        sep="\t",
        index_col=1,
    )

    gene_ = hgnc.index.to_numpy()

    is_ = full(
        gene_.shape,
        True,
    )

    for (
        label,
        selection,
    ) in selection.items():

        print("Selecting by {}...".format(label))

        is_ &= asarray(
            tuple(
                isinstance(
                    value,
                    str,
                )
                and value in selection
                for value in hgnc.loc[
                    :,
                    label,
                ].to_numpy()
            ),
        )

        print(
            "{}/{}".format(
                is_.sum(),
                is_.size,
            )
        )

    return gene_[is_]
