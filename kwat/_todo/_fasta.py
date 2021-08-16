from os.path import isfile

from pandas import read_csv

from .support import command


def get_chromosome_size_from_fasta_gz(
    fasta_gz_file_path,
):

    return read_csv(
        "{}.fai".format(fasta_gz_file_path),
        "\t",
        header=None,
        usecols=(0, 1),
        index_col=0,
        squeeze=True,
    ).to_dict()


def get_sequence_from_fasta_or_fasta_gz(
    fasta_or_fasta_gz_file_path,
    chromosome,
    _1_indexed_inclusive_start_position,
    _1_indexed_inclusive_end_position,
):

    if not isfile("{}.gzi".format(fasta_or_fasta_gz_file_path)):

        command("samtools faidx {}".format(fasta_or_fasta_gz_file_path))

    # TODO: try [:-1] instead of strip
    return "".join(
        command(
            "samtools faidx {} {}:{}-{}".format(
                fasta_or_fasta_gz_file_path,
                chromosome,
                _1_indexed_inclusive_start_position,
                _1_indexed_inclusive_end_position,
            )
        )
        .stdout.strip()
        .splitlines[1:]
    )
