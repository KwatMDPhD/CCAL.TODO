from pandas import read_csv

from .DATA_DIRECTORY_PATH import DATA_DIRECTORY_PATH

refseq_nm_nc_gene = read_csv(
    "{}/refseq_nm_nc_gene.tsv.gz".format(DATA_DIRECTORY_PATH), sep="\t"
)

REFSEQ_GENE = {}

for column in ("RefSeq mRNA ID", "RefSeq ncRNA ID"):

    refseq_gene = refseq_nm_nc_gene[[column, "Gene name"]].dropna()

    REFSEQ_GENE.update(dict(zip(refseq_gene.iloc[:, 0], refseq_gene.iloc[:, 1])))
