from numpy import asarray, full, unique
from pandas import isna, read_csv

from .CONSTANT import DATA_DIRECTORY_PATH


def get_gene_symbol(column_to_selections=None):

    if column_to_selections is None:

        column_to_selections = {"Locus group": ("protein-coding gene",)}

    table = read_csv("{}/hgnc_gene_group.tsv.gz".format(DATA_DIRECTORY_PATH), sep="\t")

    is_selected = full(table.shape[0], True)

    for label, selections in column_to_selections.items():

        print(label)

        is_selected &= asarray(
            tuple(
                not isna(object_) and object_ in selections
                for object_ in table.loc[:, label].to_numpy()
            )
        )

        print("{}/{}".format(is_selected.sum(), is_selected.size))

    genes = (
        table.loc[
            is_selected, ["symbol" in label for label in table.columns.to_numpy()],
        ]
        .to_numpy()
        .ravel()
    )

    return unique(genes[~isna(genes)])


enst_ensg_gene = read_csv(
    "{}/enst_ensg_gene.tsv.gz".format(DATA_DIRECTORY_PATH), sep="\t"
)

ENS_GENE = {}

for column in ("Transcript stable ID version", "Gene stable ID version"):

    ens_gene = enst_ensg_gene[[column, "Gene name"]].dropna()

    ENS_GENE.update(dict(zip(ens_gene.iloc[:, 0], ens_gene.iloc[:, 1])))

for ens, gene in tuple(ENS_GENE.items()):

    ens = ens.split(sep=".")[0]

    ENS_GENE[ens] = gene

    ENS_GENE["{}_at".format(ens)] = gene


refseq_nm_nc_gene = read_csv(
    "{}/refseq_nm_nc_gene.tsv.gz".format(DATA_DIRECTORY_PATH), sep="\t"
)

REFSEQ_GENE = {}

for column in ("RefSeq mRNA ID", "RefSeq ncRNA ID"):

    refseq_gene = refseq_nm_nc_gene[[column, "Gene name"]].dropna()

    REFSEQ_GENE.update(dict(zip(refseq_gene.iloc[:, 0], refseq_gene.iloc[:, 1])))


ILMNID_GENE = (
    read_csv(
        "{}/HumanMethylation450_15017482_v1-2.csv.gz".format(DATA_DIRECTORY_PATH),
        skiprows=7,
        usecols=(0, 21,),
        index_col=0,
        squeeze=True,
    )
    .dropna()
    .apply(lambda str_: str_.split(";")[0])
    .to_dict()
)


def map_genes(ids):

    n_gene_genes = {}

    for dict_ in (ENS_GENE, ILMNID_GENE, REFSEQ_GENE):

        genes = tuple(map(dict_.get, ids))

        genes_is_None = tuple(gene is None for gene in genes)

        if not all(genes_is_None):

            n_gene = len(genes_is_None) - sum(genes_is_None)

            print("Mapped {} genes.".format(n_gene))

            n_gene_genes[n_gene] = genes

    if 0 < len(n_gene_genes):

        return n_gene_genes[max(n_gene_genes.keys())]


def rename_cell_lines(names):

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

        print("Failed {}.".format(sorted(set(fails))))

    return tuple(renames)
