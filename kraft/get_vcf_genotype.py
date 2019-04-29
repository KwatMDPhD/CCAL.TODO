def get_vcf_genotype(ref, alt, gt):

    return [
        ([ref] + alt.split(sep=","))[int(i)]
        for i in gt.replace("/", "|").split(sep="|")
    ]
