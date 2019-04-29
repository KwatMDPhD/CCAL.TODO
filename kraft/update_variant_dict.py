from .get_variant_start_and_end_positions import get_variant_start_and_end_positions
from .get_vcf_allelic_frequencies import get_vcf_allelic_frequencies
from .get_vcf_genotype import get_vcf_genotype
from .get_vcf_population_allelic_frequencies import (
    get_vcf_population_allelic_frequencies,
)


def update_variant_dict(variant_dict):

    ref = variant_dict["REF"]

    alt = variant_dict["ALT"]

    start_position, end_position = get_variant_start_and_end_positions(
        int(variant_dict["POS"]), ref, alt
    )

    variant_dict["start_position"] = start_position

    variant_dict["end_position"] = end_position

    caf = variant_dict.get("CAF")

    if caf:

        variant_dict[
            "population_allelic_frequencies"
        ] = get_vcf_population_allelic_frequencies(caf)

    for sample_dict in variant_dict["sample"].values():

        if "GT" in sample_dict:

            sample_dict["genotype"] = get_vcf_genotype(ref, alt, sample_dict["GT"])

        if "AD" in sample_dict and "DP" in sample_dict:

            sample_dict["allelic_frequency"] = get_vcf_allelic_frequencies(
                sample_dict["AD"], sample_dict["DP"]
            )
