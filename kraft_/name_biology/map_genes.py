from .ENS_GENE import ENS_GENE
from .ILMNID_GENE import ILMNID_GENE
from .REFSEQ_GENE import REFSEQ_GENE


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
