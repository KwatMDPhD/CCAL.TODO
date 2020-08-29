def read(file_path):

    set_to_gene_ = {}

    with open(file_path) as io:

        for line in io.readlines():

            split_ = line.split(sep="\t")

            set_to_gene_[split_[0]] = tuple(gene for gene in split_[2:-1] if gene != "")

    return set_to_gene_


def read_multiple(file_path_):

    gene_set_gene_ = {}

    for file_path in file_path_:

        gene_set_gene_.update(read(file_path))

    return gene_set_gene_
