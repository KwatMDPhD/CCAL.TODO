from tabix import open as tabix_open

from .make_variant_dict_from_vcf_row import make_variant_dict_from_vcf_row
from .update_variant_dict import update_variant_dict


def get_vcf_variants_by_region(
    vcf_or_vcf_gz_file_path_or_pytabix_handle, chromosome, start_position, end_position
):

    if isinstance(vcf_or_vcf_gz_file_path_or_pytabix_handle, str):

        pytabix_handle = tabix_open(vcf_or_vcf_gz_file_path_or_pytabix_handle)

    else:

        pytabix_handle = vcf_or_vcf_gz_file_path_or_pytabix_handle

    variants = pytabix_handle.query(chromosome, start_position, end_position)

    varinat_dicts = [make_variant_dict_from_vcf_row(variant) for variant in variants]

    for variant_dict in varinat_dicts:

        update_variant_dict(variant_dict)

    return varinat_dicts
