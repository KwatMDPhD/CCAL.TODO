def read_gmt(gmt_file_path):

    gene_set_genes = {}

    with open(gmt_file_path, "r") as io:

        for line in io.readlines():

            line_split = [str_ for str_ in line.strip().split("\t") if str_]

            gene_set_genes[line_split[0]] = line_split[2:]

    return gene_set_genes
