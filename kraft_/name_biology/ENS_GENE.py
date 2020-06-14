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
