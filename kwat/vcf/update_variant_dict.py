from .get_variant_start_and_end_positions import get_variant_start_and_end_positions
from .get_vcf_genotype import get_vcf_genotype


def update_variant_dict(vd):

    re = vd["REF"]

    al = vd["ALT"]

    st, en = get_variant_start_and_end_positions(int(vd["POS"]), re, al)

    vd["start_position"] = st

    vd["end_position"] = en

    if "CAF" in vd:

        vd["population_allelic_frequencies"] = [
            float(caf_) for caf_ in vd["CAF"].split(sep=",")
        ]

    for sd in vd["sample"].values():

        if "GT" in sd:

            sd["genotype"] = get_vcf_genotype(re, al, sd["GT"])

        if "AD" in sd and "DP" in sd:

            sd["allelic_frequency"] = [
                int(ad_) / int(sd["DP"]) for ad_ in sd["AD"].split(sep=",")
            ]
