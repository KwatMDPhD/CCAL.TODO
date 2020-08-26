def read(gmt_file_path):

    set_to_genes = {}

    with open(gmt_file_path) as io:

        for line in io.readlines():

            splits = line.split(sep="\t")

            set_to_genes[splits[0]] = tuple(gene for gene in splits[2:-1] if gene != "")

    return set_to_genes


def read_many(gmt_file_paths):

    gene_set_genes = {}

    for file_path in gmt_file_paths:

        gene_set_genes.update(read(file_path))

    return gene_set_genes
