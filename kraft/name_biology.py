from pandas import read_csv

from .DATA_DIRECTORY_PATH import DATA_DIRECTORY_PATH

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


def map_genes(strs):

    n_gene_genes = {}

    for dict_ in (ENS_GENE, ILMNID_GENE, REFSEQ_GENE):

        genes = tuple(map(dict_.get, strs))

        genes_is_None = tuple(gene is None for gene in genes)

        if not all(genes_is_None):

            n_gene = len(genes_is_None) - sum(genes_is_None)

            print("Mapped {} genes.".format(n_gene))

            n_gene_genes[n_gene] = genes

    if 0 < len(n_gene_genes):

        return n_gene_genes[max(n_gene_genes.keys())]
