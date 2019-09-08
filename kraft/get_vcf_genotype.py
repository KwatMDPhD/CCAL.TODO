def get_vcf_genotype(ref, alt, gt):

    genotypes = (ref, *alt.split(sep=","))

    return tuple(genotypes[int(i)] for i in gt.replace("/", "|").split(sep="|"))
