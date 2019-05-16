from os.path import isfile
from .run_command import run_command


def get_variants_from_vcf_or_vcf_gz(
    vcf_or_vcf_gz_file_path,
    chromosome,
    _1_indexed_inclusive_start_position,
    _1_indexed_inclusive_end_position,
):

    if not isfile(f"{vcf_or_vcf_gz_file_path}.tbi"):

        run_command(f"tabix {vcf_or_vcf_gz_file_path}")

    return [
        vcf_row.split("\t")
        for vcf_row in run_command(
            f"tabix {vcf_or_vcf_gz_file_path} {chromosome}:{_1_indexed_inclusive_start_position}-{_1_indexed_inclusive_end_position}"
        )
        .stdout.strip()
        .split("\n")
    ]
