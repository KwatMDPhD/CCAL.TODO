from os.path import isfile
from .run_command import run_command


def get_sequence_from_fasta_or_fasta_gz(
    fasta_or_fasta_gz_file_path,
    chromosome,
    _1_indexed_inclusive_start_position,
    _1_indexed_inclusive_end_position,
):

    if not isfile(f"{fasta_or_fasta_gz_file_path}.gzi"):

        run_command(f"samtools faidx {fasta_or_fasta_gz_file_path}")

    return "".join(
        run_command(
            f"samtools faidx {fasta_or_fasta_gz_file_path} {chromosome}:{_1_indexed_inclusive_start_position}-{_1_indexed_inclusive_end_position}"
        )
        .stdout.strip()
        .split(sep="\n")[1:]
    )
