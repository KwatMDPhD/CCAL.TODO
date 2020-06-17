from json import dump, load

from pandas import read_csv


def read_json(json_file_path):

    with open(json_file_path) as io:

        return load(io)


def write_json(json_file_path, dict_, indent=2):

    with open(json_file_path, mode="w") as io:

        dump(dict_, io, indent=indent)


def read_gct(gct_file_path):

    return read_csv(gct_file_path, skiprows=2, sep="\t", index_col=0).drop(
        "Description", axis=1
    )


def read_gmt(gmt_file_path):

    gene_set_genes = {}

    with open(gmt_file_path) as io:

        for line in io.readlines():

            line_split = tuple(str_ for str_ in line.strip().split("\t") if str_)

            gene_set_genes[line_split[0]] = line_split[2:]

    return gene_set_genes
