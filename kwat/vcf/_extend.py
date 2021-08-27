from ._get_genotype import _get_genotype
from ._get_variant_start_and_end import _get_variant_start_and_end


def _extend(vd):

    re = vd["REF"]

    al = vd["ALT"]

    st, en = _get_variant_start_and_end(int(vd["POS"]), re, al)

    vd["start_position"] = st

    vd["end_position"] = en

    if "CAF" in vd:

        vd["population_allelic_frequencies"] = [
            float(ca) for ca in vd["CAF"].split(sep=",")
        ]

    for sa in vd["sample"].values():

        if "GT" in sa:

            sa["genotype"] = _get_genotype(re, al, sa["GT"])

        if "AD" in sa and "DP" in sa:

            sa["allelic_frequency"] = [
                int(ad) / int(sa["DP"]) for ad in sa["AD"].split(sep=",")
            ]
