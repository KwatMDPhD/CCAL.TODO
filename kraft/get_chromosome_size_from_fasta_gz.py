from pandas import read_csv


def get_chromosome_size_from_fasta_gz(fasta_gz_file_path):

    return read_csv(
        "{}.fai".format(fasta_gz_file_path),
        sep="\t",
        header=None,
        usecols=(0, 1),
        index_col=0,
        squeeze=True,
    ).to_dict()
