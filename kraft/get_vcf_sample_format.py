def get_vcf_sample_format(format_, key, sample):

    return sample.split(sep=":")[format_.split(sep=":").index(key)]
