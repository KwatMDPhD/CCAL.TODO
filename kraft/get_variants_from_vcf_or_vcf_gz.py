from os.path import isfile

from .run_command import run_command


def get_variants_from_vcf_or_vcf_gz(
    vcf_or_vcf_gz_file_path,
    chromosome,
    _1_indexed_inclusive_start_position,
    _1_indexed_inclusive_end_position,
):

    if not isfile("{}.tbi".format(vcf_or_vcf_gz_file_path)):

        run_command("tabix {}".format(vcf_or_vcf_gz_file_path))

    return tuple(
        tuple(vcf_row.split(sep="\t"))
        for vcf_row in run_command(
            "tabix {} {}:{}-{}".format(
                vcf_or_vcf_gz_file_path,
                chromosome,
                _1_indexed_inclusive_start_position,
                _1_indexed_inclusive_end_position,
            )
        )
        .stdout.strip()
        .split(sep="\n")
        if vcf_row != ""
    )
