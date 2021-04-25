def _read(path):

    set_to_gene_ = {}

    with open(path) as io:

        for line in io.readlines():

            split_ = line.strip().split(sep="\t")

            set_to_gene_[split_[0]] = tuple(gene for gene in split_[2:] if gene != "")

    return set_to_gene_


def read(path_):

    if isinstance(path_, str):

        path_ = (path_,)

    set_to_gene_ = {}

    for path in path_:

        set_to_gene_.update(_read(path))

    return set_to_gene_
